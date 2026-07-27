import torch
import torchaudio
import os
import argparse
import sys
import librosa 
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from audiocraft.data.audio import audio_read
from torch.nn import functional as F
import numpy as np
import pandas as pd

def setup_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return "cpu"
    return "cuda"

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model = model.to(device)
    model.eval()
    
    compression_ratio = model.config.audio_encoder.frame_rate / model.config.audio_encoder.sampling_rate
    print(f"Compression ratio: {compression_ratio:.4f} ({model.config.audio_encoder.frame_rate} tokens per second of audio)")
    return model, processor

def _compute_cross_entropy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, reduction="mean"
    ):
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    
    if reduction == "none":
        # return per-codebook, per-step loss of shape (B, K, T)
        ce_per_token = torch.zeros_like(targets, dtype=torch.float32)
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            
            ce_flat = F.cross_entropy(logits_k, targets_k, reduction="none")
            ce_masked = ce_flat * mask_k.float()
            ce_per_token[:, k, :] = ce_masked.view(B, T)
        
        return ce_per_token
    else:
        # original mean-reduction logic
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        ce = ce / K
        return ce, ce_per_codebook

def save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path=None):
    """
    Save per-token losses to CSV (average across codebooks only)
    """
    if relative_path:
        # use the relative path, replacing separators with underscores for filesystem safety
        rel_dir = os.path.dirname(relative_path)
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        if rel_dir:
            # replace path separators with underscores to build a unique filename
            filename = f"{rel_dir.replace(os.sep, '_')}_{base_filename}"
        else:
            filename = base_filename
    else:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    avg_csv_file = os.path.join(token_output_dir, f"{filename}_tokens_avg.csv")
    os.makedirs(token_output_dir, exist_ok=True)
    
    try:
        # convert to numpy array
        ce_np = ce_per_token.cpu().numpy()
        
        # compute the mean at each token position
        avg_records = []
        
        if ce_np.ndim == 3:  # Shape: (B, K, T)
            B, K, T = ce_np.shape
            for b in range(B):
                for t in range(T):
                    # average over all codebooks at this token position
                    token_avg = float(ce_np[b, :, t].mean())
                    avg_records.append({
                        'token_position': t,
                        'avg_loss_value': token_avg
                    })
        elif ce_np.ndim == 2:  # Shape: (K, T)
            K, T = ce_np.shape
            for t in range(T):
                # average over all codebooks at this token position
                token_avg = float(ce_np[:, t].mean())
                avg_records.append({
                    'token_position': t,
                    'avg_loss_value': token_avg
                })
        
        # save the averaged data
        if avg_records:
            avg_df = pd.DataFrame(avg_records)
            avg_df.to_csv(avg_csv_file, index=False)
            print(f"Saved {len(avg_records)} averaged records to {avg_csv_file}")
            
            # compute and return the overall mean loss
            overall_mean = avg_df['avg_loss_value'].mean()
            print(f"Overall mean loss: {overall_mean:.6f}")
            
            return overall_mean
        else:
            print(f"No data to save: {filename}")
            return None
            
    except Exception as e:
        print(f"Error saving CSV {filename}: {str(e)}")
        return None

def process_single_audio(audio_path, model, processor, device, reduction="mean", token_output_dir=None, relative_path=None):
    try:
        if device == "cuda":
            audio, sr = audio_read(audio_path)
            audio = audio.mean(dim=0, keepdim=True)
            if sr != 32000:
                audio = torchaudio.functional.resample(audio, sr, 32000)
            audio = audio.to(device)
            valid_token_length = min(audio.shape[-1] // 320, 1500)
        else:
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
            audio = torch.tensor(audio).unsqueeze(0).to(device)
            valid_token_length = min(audio.shape[-1] // 320, 1500)

        with torch.no_grad():
            encoded = model.audio_encoder.encode(audio.unsqueeze(0)) 
            tokens = encoded.audio_codes.long()

        B, C, K, T = tokens.shape
        input_tokens = tokens[:, :, :, :-1].contiguous()
        target_tokens = tokens[:, :, :, 1:].contiguous()

        mask = torch.zeros_like(target_tokens, dtype=torch.bool)
        for b in range(B):
            for c in range(C):
                mask[b, c, :, :valid_token_length] = True

        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                input_tokens_reshaped = input_tokens.view(B, C * K, -1)
                model.decoder = model.decoder.to(dtype=torch.float32)
                outputs = model.decoder(input_tokens_reshaped)
                output_logits = outputs.logits

            logits = output_logits.view(B, C, K, -1, output_logits.size(-1))
            
            if reduction == "none":
                # compute per-token loss
                ce_per_token = _compute_cross_entropy(
                    logits.view(B * C, K, -1, output_logits.size(-1)), 
                    target_tokens.view(B * C, K, -1), 
                    mask.view(B * C, K, -1),
                    reduction=reduction)
                
                if token_output_dir:
                    # save CSV and return the mean loss
                    avg_loss = save_tokens_as_csv(ce_per_token, audio_path, token_output_dir, relative_path)
                    return avg_loss if avg_loss is not None else ce_per_token.mean().item()
                else:
                    return ce_per_token.mean().item()
            else:
                # compute mean loss
                ce, ce_per_codebook = _compute_cross_entropy(
                    logits.view(B * C, K, -1, output_logits.size(-1)), 
                    target_tokens.view(B * C, K, -1), 
                    mask.view(B * C, K, -1),
                    reduction=reduction)
                return ce.item()

    except Exception as e:
        print(f"Error processing audio {os.path.basename(audio_path)}: {str(e)}")
        return None

def process_audio_directory(audio_dir, model, processor, device, output_file=None, reduction="mean", token_output_dir=None):
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                rel_path = os.path.relpath(os.path.join(root, f), audio_dir)
                audio_files.append(rel_path)
    
    if not audio_files:
        raise ValueError(f"No WAV/MP3 files found in {audio_dir}")

    processed_files = set()
    if output_file and os.path.exists(output_file):
        print(f"Resuming from existing results file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                if ':' in line:
                    filename = line.split(':', 1)[0].strip()
                    processed_files.add(filename)
        print(f"Found {len(processed_files)} already processed files.")

    files_to_process = [f for f in audio_files if f not in processed_files]
    if not files_to_process:
        print("All files have already been processed. Nothing to do.")
        return [], []
    
    print(f"Total files: {len(audio_files)}. Already processed: {len(processed_files)}. To process: {len(files_to_process)}")

    results = []
    error_files = [] 

    out_file = None
    if output_file:
        out_file = open(output_file, 'a') # Open in append mode
        print(f"Results will be appended to: {output_file}")

    for rel_filename in tqdm(files_to_process, desc="Processing"):
        audio_path = os.path.join(audio_dir, rel_filename)
        loss = process_single_audio(audio_path, model, processor, device, reduction, token_output_dir, rel_filename)
        if loss is not None:
            results.append((rel_filename, loss))
            if out_file:
                out_file.write(f"{rel_filename}: {loss:.8f}\n")
                out_file.flush()
        else:
            error_files.append(rel_filename)

    if out_file:
        out_file.close()

    if error_files and device == "cuda":
        error_file_path = output_file + ".errors.txt" if output_file else "processing_errors.txt"
        with open(error_file_path, 'a') as f: # Append to error file too
            f.write(f"--- New run ---\n")
            for filename in error_files:
                f.write(f"{filename}\n")
        print(f"\n{len(error_files)} failed files appended to: {error_file_path}")

    return results, error_files

def main():
    parser = argparse.ArgumentParser(description="Compute audio loss values with the musicgen-small model")
    parser.add_argument("--audio_dir", type=str, required=True, help="directory of audio files")
    parser.add_argument("--output_file", type=str, default="results_small.txt", help="output results file path")
    parser.add_argument("--force_cpu", action="store_true", help="force CPU for all files")
    parser.add_argument("--reduction", type=str, choices=["mean", "none"], default="mean", help="loss reduction: mean or none (none saves per-token losses)")
    parser.add_argument("--token_output_dir", type=str, default=None, help="directory for per-token losses when reduction=none")
    
    args = parser.parse_args()

    if args.reduction == "none" and not args.token_output_dir:
        args.token_output_dir = "token_losses"
        print(f"Using default token output dir: {args.token_output_dir}")

    original_stdout = None
    if args.output_file:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_file + ".log", 'w')

    device = setup_device(args.force_cpu)
    print(f"Using device: {device}")

    model, processor = load_model(device)

    assert os.path.isdir(args.audio_dir), f"Directory not found: {args.audio_dir}"

    results, error_files = process_audio_directory(
        args.audio_dir, model, processor, device, args.output_file, args.reduction, args.token_output_dir
    )

    if not args.output_file:
        print("\nResults:")
        for filename, loss in results:
            print(f"{filename}: {loss:.4f}")

    if 'device' in locals() and device == 'cuda':
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("GPU memory cleared")
        except Exception as e:
            print(f"Failed to clear GPU memory: {e}")
    else:
        print("GPU memory cleared")

    if original_stdout:
        sys.stdout.close()
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()