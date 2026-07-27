"""MusicGen delay-pattern teacher forcing (HF transformers path).

Background: a naive teacher-forcing forward feeds the time-aligned codes
directly to ``model.decoder`` ("parallel" protocol), without the delayed
codebook interleaving MusicGen was trained with. That protocol is internally
consistent but is not the model's true likelihood, and it systematically
inflates losses across model scales.

This module provides the delay-correct forward: inputs are arranged with the
MusicGen-native delay pattern (codebook k shifted right by k steps, leading
positions filled with the pad/special token), and after a single forward pass
each codebook's logits are re-aligned to the audio-frame timeline — matching
audiocraft's ``LMModel.compute_predictions``.
"""
from __future__ import annotations

import torch


def delayed_aligned_logits(model, codes: torch.Tensor) -> torch.Tensor:
    """codes ``[1, C, K, T]`` → frame-aligned logits ``[C, K, T-1, V]``.

    Column t of the output (t = 0..T-2) holds the logits for "predict audio
    frame t+1 given the delayed-interleaved context" — the same frame indexing
    as the parallel protocol, so the two are drop-in interchangeable.
    Coverage: S-1 = T+K-2 ≥ T+k-1 for all k < K, so every frame of every
    codebook has a prediction; there are no missing positions.
    """
    pad_id = int(model.decoder.config.pad_token_id)
    bsz, channels, K, T = codes.shape
    assert bsz == 1, "delayed_aligned_logits supports bsz=1 only (matches the extraction loop)"
    S = T + K - 1
    outs = []
    for ch in range(channels):
        c = codes[0, ch]  # [K, T]
        dl = torch.full((K, S), pad_id, dtype=torch.long, device=c.device)
        for k in range(K):
            dl[k, k : k + T] = c[k]
        lg = model.decoder(dl[None, :, :-1], output_hidden_states=False).logits
        if lg.dim() == 4:
            lg = lg[0]  # [K, S-1, V]
        aligned = torch.empty((K, T - 1, lg.shape[-1]), dtype=lg.dtype, device=lg.device)
        for k in range(K):
            # seq position s predicts dl[k, s+1] = audio frame s+1-k;
            # frames 1..T-1 correspond to s = frame + k - 1
            aligned[k] = lg[k, k : k + T - 1, :]
        outs.append(aligned)
    return torch.stack(outs, dim=0)  # [C, K, T-1, V]
