"""Direction 1 (step A): extract temporal embeddings for spectral probing.

Default mode now follows the user-defined setting:
- musicgen_lastlayer: MusicGen-small decoder last-layer token embeddings.

Compatibility mode:
- cnn_proxy: LossCurveCNN conv feature map proxy (legacy).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import torch
from tqdm import tqdm
from transformers import MusicgenForConditionalGeneration

from _shared import MAX_LEN, get_split_df, load_loss_curve, load_model, make_model_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-name", type=str, default="loss_curve_cnn")
    parser.add_argument(
        "--embedding-source",
        type=str,
        default="musicgen_lastlayer",
        choices=["musicgen_lastlayer", "cnn_proxy"],
        help="musicgen_lastlayer is the primary intended setup.",
    )
    parser.add_argument("--musicgen-model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all.")
    parser.add_argument(
        "--fixed-time-steps",
        type=int,
        default=0,
        help=(
            "If >0, pad (zero) or truncate each embedding along time to this length before saving. "
            "Use the same value for all runs when doing spectral probing so FFT bands align across clips. "
            "0 = keep native length (legacy). Typical: match loss-curve MAX_LEN (1500) from _shared.py."
        ),
    )
    parser.add_argument(
        "--save-entropy-curve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="musicgen_lastlayer only: save per-step entropy (mean over codebooks) from decoder logits.",
    )
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "outputs"))
    return parser.parse_args()


def sample_uid(token_loss_path: str) -> str:
    return hashlib.md5(token_loss_path.encode("utf-8")).hexdigest()[:16]


def fix_temporal_length(emb: np.ndarray, fixed_t: int) -> tuple[np.ndarray, int]:
    """Pad with zeros at the end or truncate so time dimension equals fixed_t.

    Returns (emb_out, valid_time_steps) where valid_time_steps is the original T before pad/truncate.
    """
    if fixed_t <= 0:
        return emb, int(emb.shape[0])
    t0, d = int(emb.shape[0]), int(emb.shape[1])
    if t0 >= fixed_t:
        return emb[:fixed_t].astype(np.float32, copy=True), t0
    out = np.zeros((fixed_t, d), dtype=np.float32)
    out[:t0] = emb.astype(np.float32, copy=False)
    return out, t0


def fix_temporal_length_1d(h: np.ndarray, fixed_t: int) -> tuple[np.ndarray, int]:
    """Same length policy as fix_temporal_length; pad tail with NaN (invalid) for 1D curves."""
    if fixed_t <= 0:
        return h.astype(np.float32, copy=False), int(h.shape[0])
    t0 = int(h.shape[0])
    if t0 >= fixed_t:
        return h[:fixed_t].astype(np.float32, copy=True), t0
    out = np.full((fixed_t,), np.nan, dtype=np.float32)
    out[:t0] = h.astype(np.float32, copy=False)
    return out, t0


def entropy_from_logits_multi_codebook(logits: torch.Tensor) -> torch.Tensor:
    """logits: [K, T, V] -> per-step mean entropy [T] (natural log, nats)."""
    p = torch.softmax(logits.float(), dim=-1)
    log_p = torch.log(p.clamp_min(1e-12))
    h_kt = -(p * log_p).sum(dim=-1)  # [K, T]
    return h_kt.mean(dim=0)


def extract_embedding_proxy(model: torch.nn.Module, x: torch.Tensor, valid_len: int) -> np.ndarray:
    """Return (T', D) temporal embeddings from CNN conv blocks.

    Current proxy:
    x -> conv1/relu/pool -> conv2/relu  (before second pool) => [1, 32, 750]
    """
    z = x
    for idx in [0, 1, 2, 3, 4]:
        z = model.features[idx](z)
    z = z[0].detach().cpu().numpy()  # [C, T']
    emb = z.T.astype(np.float32)  # [T', C]

    valid_t = max(1, int(np.ceil(valid_len / 2.0)))
    emb = emb[:valid_t]
    return emb


def load_audio_mono_32k(audio_path: str, device: torch.device) -> torch.Tensor:
    audio, sr = sf.read(audio_path)
    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T
    wav = torch.from_numpy(audio).float()
    if sr != 32000:
        wav = torchaudio.functional.resample(wav, sr, 32000)
    wav = wav.mean(dim=0, keepdim=True)  # [1, T]
    return wav.to(device)


def extract_musicgen_lastlayer_embedding(
    musicgen_model: MusicgenForConditionalGeneration,
    audio_path: str,
    device: torch.device,
    *,
    return_entropy: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    wav = load_audio_mono_32k(audio_path, device=device)
    with torch.no_grad():
        enc = musicgen_model.audio_encoder.encode(wav.unsqueeze(0))
        codes = enc.audio_codes.long()
        bsz, c, k, t = codes.shape
        inp = codes[:, :, :, :-1].contiguous().view(bsz, c * k, t - 1)
        out = musicgen_model.decoder(inp, output_hidden_states=True)
        hidden = out.hidden_states[-1].squeeze(0)  # [T, 1024]
        emb = hidden.detach().cpu().numpy().astype(np.float32)
        if not return_entropy:
            return emb
        logits = out.logits
        if logits.dim() == 4:
            logits = logits.squeeze(0)
        ent_t = entropy_from_logits_multi_codebook(logits).detach().cpu().numpy().astype(np.float32)
        return emb, ent_t


def main() -> None:
    args = parse_args()
    subdir = f"{args.splits}_{args.split}_{args.embedding_source}"
    if args.fixed_time_steps > 0:
        subdir = f"{subdir}_T{args.fixed_time_steps}"
    base_out = os.path.join(args.out_dir, "embeddings", subdir)
    os.makedirs(base_out, exist_ok=True)

    df = get_split_df(args.splits, args.split)
    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = None
    musicgen_model = None
    if args.embedding_source == "cnn_proxy":
        cnn_model = load_model(args.run_name, device=device, max_len=MAX_LEN)
    else:
        musicgen_model = MusicgenForConditionalGeneration.from_pretrained(args.musicgen_model).to(device)
        musicgen_model.eval()

    records: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract-embeddings"):
        ent_native: np.ndarray | None = None
        if args.embedding_source == "cnn_proxy":
            curve, mask, valid_len = load_loss_curve(row["token_loss_path"], max_len=MAX_LEN)
            x = make_model_input(curve, mask, device=device)
            assert cnn_model is not None
            with torch.no_grad():
                emb = extract_embedding_proxy(cnn_model, x, valid_len=valid_len)
            emb_src = "loss_curve_cnn_conv2_proxy"
        else:
            assert musicgen_model is not None
            if args.save_entropy_curve:
                emb, ent_native = extract_musicgen_lastlayer_embedding(
                    musicgen_model=musicgen_model,
                    audio_path=row["audio_path"],
                    device=device,
                    return_entropy=True,
                )
            else:
                emb = extract_musicgen_lastlayer_embedding(
                    musicgen_model=musicgen_model,
                    audio_path=row["audio_path"],
                    device=device,
                    return_entropy=False,
                )
            emb_src = "musicgen_small_decoder_lastlayer"

        emb, native_t = fix_temporal_length(emb, args.fixed_time_steps)
        uid = sample_uid(row["token_loss_path"])
        entropy_path = ""
        mean_entropy = float("nan")
        if args.embedding_source == "musicgen_lastlayer" and args.save_entropy_curve and ent_native is not None:
            ent_pad, _ = fix_temporal_length_1d(ent_native, args.fixed_time_steps)
            entropy_path = os.path.abspath(os.path.join(base_out, f"{uid}_entropy.npy"))
            np.save(entropy_path, ent_pad)
            mean_entropy = float(np.nanmean(ent_pad))

        emb_path = os.path.abspath(os.path.join(base_out, f"{uid}.npy"))
        np.save(emb_path, emb)

        rec = {
            "audio_path": row["audio_path"],
            "token_loss_path": row["token_loss_path"],
            "target_score": float(row["score"]),
            "embedding_path": emb_path,
            "entropy_curve_path": entropy_path,
            "mean_entropy": mean_entropy,
            "time_steps": int(emb.shape[0]),
            "valid_time_steps": int(native_t),
            "fixed_time_steps": int(args.fixed_time_steps) if args.fixed_time_steps > 0 else "",
            "dim": int(emb.shape[1]),
            "embedding_source": emb_src,
        }
        records.append(rec)

    manifest = pd.DataFrame(records)
    manifest_suffix = args.embedding_source
    if args.fixed_time_steps > 0:
        manifest_suffix = f"{manifest_suffix}_T{args.fixed_time_steps}"
    manifest_path = os.path.join(
        args.out_dir, f"embedding_manifest_{args.splits}_{args.split}_{manifest_suffix}.csv"
    )
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path} ({len(manifest)} rows).")


if __name__ == "__main__":
    main()
