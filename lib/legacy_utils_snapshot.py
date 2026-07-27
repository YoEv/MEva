import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SAEDataset(Dataset):
    """
    Common Dataset class for SAE features and MOS scores.
    """
    def __init__(self, features_dict, mos_data, return_fname=False):
        self.samples = []
        self.return_fname = return_fname
        common = sorted(list(set(features_dict.keys()) & set(mos_data.keys())))
        
        for fname in common:
            feat = features_dict[fname]
            score = np.mean(mos_data[fname])
            self.samples.append((feat, score, fname))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y, fname = self.samples[idx]
        if self.return_fname:
            return x, torch.tensor(y, dtype=torch.float32), fname
        else:
            return x, torch.tensor(y, dtype=torch.float32)

def extract_sae_sequence(sae, model, audio_path, target_len=1500):
    """
    Extracts SAE features as a time sequence.
    Common utility for both Training and Inference/Interpretation.
    """
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim == 1: audio = audio[None, :]
        else: audio = audio.T
        wav = torch.from_numpy(audio).float()
        if sr != 32000:
            wav = torchaudio.functional.resample(wav, sr, 32000)
        wav = wav.mean(dim=0, keepdim=True).to(DEVICE)
        
        with torch.no_grad():
            enc = model.audio_encoder.encode(wav.unsqueeze(0))
            codes = enc.audio_codes.long()
            B, C, K, T = codes.shape
            inp = codes[:, :, :, :-1].contiguous().view(B, C*K, T-1)
            
            out = model.decoder(inp, output_hidden_states=True)
            hidden = out.hidden_states[-1].squeeze(0)
            
            # SAE Forward
            # SAE model now returns (x_hat, f), we only need f
            _, f = sae(hidden) # [T, 16384]
            
            # Interpolate to target_len to normalize input size
            # [1, T, 16384] -> [1, 16384, T] for interpolate
            f = f.unsqueeze(0).permute(0, 2, 1)
            f = torch.nn.functional.interpolate(f, size=target_len, mode='linear', align_corners=False)
            f = f.permute(0, 2, 1).squeeze(0) # [1500, 16384]
            
            return f.cpu()
            
    except Exception as e:
        print(f"Error extracting {audio_path}: {e}")
        return None

def load_raw_mos(path):
    """
    Loads raw MOS scores.
    Returns: dict {filename: [score1, score2, ...]}
    """
    print(f"Loading Raw MOS from {path}...")
    scores = defaultdict(list)
    if not os.path.exists(path):
        print(f"Error: MOS file not found at {path}")
        return {}
        
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) < 3: continue
            
            # Format: filename, rater, score1, score2, ...
            fname = parts[0].replace(".wav", "").strip()
            # Assuming all columns from index 2 are scores
            vals = [float(x) for x in parts[2:]]
            scores[fname].extend(vals)
                
    print(f"  -> Loaded raw scores for {len(scores)} files.")
    return scores

def load_metrics(path, metric_cols=None):
    """
    Loads metrics from a tab-separated file.
    Args:
        metric_cols: dict mapping {col_name: col_index}
                     If None, assumes 'filename\tscore' format.
    Returns: dict {filename: {metric_name: value}}
    """
    print(f"Loading Metrics from {path}...")
    data = {}
    if not os.path.exists(path):
        print(f"Error: Metric file not found at {path}")
        return {}

    with open(path, 'r') as f:
        header = None
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            
            if header is None:
                if parts[0].lower() == "filename":
                    header = parts
                    continue
                else:
                    # No header?
                    pass
            
            fname = parts[0].replace(".wav", "").strip()
            
            if metric_cols:
                # Use specified columns
                metrics = {}
                for m_name, m_idx in metric_cols.items():
                    if m_idx < len(parts):
                        try:
                            metrics[m_name] = float(parts[m_idx])
                        except:
                            metrics[m_name] = np.nan
                data[fname] = metrics
            else:
                # Default single score at index 1
                if len(parts) > 1:
                    try:
                        data[fname] = {"score": float(parts[1])}
                    except:
                        pass
                        
    print(f"  -> Loaded metrics for {len(data)} files.")
    return data

def normalize_data(x):
    """Min-Max normalization to [0, 1]"""
    x = np.array(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def create_hybrid_index(master_index_path, token_loss_dir, output_path):
    """
    Creates a new, clean index file for hybrid models (Exp3 and beyond).

    This function reads a master index, finds corresponding per-token loss files,
    and generates a new index containing the audio file path, the path to the
    loss file, the human score, and the length of the loss sequence.

    Args:
        master_index_path (str): Path to the master index CSV.
        token_loss_dir (str): Path to the directory containing token loss subdirectories.
        output_path (str): Path to save the newly generated index CSV.
    """
    print(f"Reading master index from: {master_index_path}")
    master_df = pd.read_csv(master_index_path)
    
    new_index_data = []
    print("Constructing paths and building new hybrid index...")

    for _, row in tqdm(master_df.iterrows(), total=len(master_df)):
        source = row['source']
        original_path = row['path']
        score = row['score']
        
        base_filename = os.path.splitext(os.path.basename(original_path))[0]
        csv_filename = f"{base_filename}_tokens_avg.csv"
        
        token_loss_subdir = ""
        if source in ['MusicPrefs', 'HumanEval']:
            if source == 'MusicPrefs':
                token_loss_subdir = "musicprefs_converted_wavs_tokens"
            else: # HumanEval
                token_loss_subdir = "wav_tokens"
        else: # Other datasets
            parent_dir_name = os.path.basename(os.path.dirname(original_path))
            token_loss_subdir = f"{parent_dir_name}_tokens"

        csv_path = os.path.join(token_loss_dir, token_loss_subdir, csv_filename)

        if os.path.exists(csv_path):
            try:
                # Check if the CSV is not empty and contains numeric data
                data = pd.read_csv(csv_path, header=0).values
                if data.size > 0 and np.issubdtype(data.dtype, np.number):
                    new_index_data.append({
                        'score': score,
                        'audio_path': original_path, # Add the original audio path
                        'token_loss_path': csv_path,
                        'length': len(data)
                    })
            except Exception:
                # Silently skip empty or malformed CSVs
                pass
    
    output_df = pd.DataFrame(new_index_data)
    
    if len(output_df) == 0:
        print(f"Warning: The new index is empty. No matching and valid loss files were found in {os.path.join(token_loss_dir, token_loss_subdir)}")
    else:
        print(f"Successfully created a new index with {len(output_df)} verified samples.")

    output_df.to_csv(output_path, index=False)
    print(f"New hybrid index saved to: {output_path}")

