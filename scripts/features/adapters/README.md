# Cross-model intrinsic-signal adapters: unified npz schema

Each new autoregressive model gets one `<model>.py` adapter that runs a single
teacher-forcing forward pass, produces loss+entropy together, and writes an npz
in the schema below. The MusicGen family does not go through an adapter (it
uses the two-pass `extract_loss_curves.py` / `extract_entropy_curves.py` flow).

## File layout

```
features/<model_tag>/<benchmark>/<split>/<uid>.npz     # uid = stable clip id, consistent with the main pipeline
features/<model_tag>/<benchmark>/manifest_<split>.csv  # audio_path, npz_path, mean_loss, mean_entropy, T_native
```

## npz contents

| key | shape / dtype | description |
|---|---|---|
| `loss` | `(K, T) float32` | per-token NLL; K = number of codebooks (K=1 for single-codebook models) |
| `entropy` | `(K, T) float32` | per-token predictive entropy (nats) |
| `mask` | `(T,) uint8` | 1 = valid frame; 0 = padding / out-of-window |
| `meta` | `() object` (json str) | see below |

Required `meta` fields: `model_tag` (e.g. `yue_s1`), `model_hf_id`, `codec`,
`frame_hz` (token frames per second; for interleaved multi-track models, the
equivalent audio frame rate), `K`, `clip_id`, `benchmark`, `window` (the
windowing parameters used), `context_limit_tokens` (measured context cliff),
`extracted_at`.

## Conventions

0. **Native-pattern discipline (highest priority)**: teacher forcing must
   arrange input tokens exactly as the model was *trained* (delay /
   interleaving / dual-track / ...), then re-align logits to audio frames.
   The "aligned codes fed directly" parallel shortcut is not allowed — it
   systematically inflates loss, and the inflation grows with model scale.
   MusicGen reference implementation: `lib/repro/musicgen_pattern.py`.
1. **T is the model-native frame count** — no pooling inside the adapter;
   the training side applies a uniform `pool_to_frames(x, 1500)` mean-pool.
2. Multi-track / interleaved models (YuE dual-track, SongGen interleave):
   align tracks on the audio timeline first, then stack into the K axis
   (track × codebook flattened); document each row's meaning in `meta`.
   Never mix different time steps into one row.
3. Chunked models: split at the model's context limit into non-overlapping
   chunks, teacher-force within each chunk, concatenate features along time
   (mark seam frames invalid in `mask` if applicable).
4. Numerics: loss/entropy in natural log; set inf/nan to 0 and zero the mask.
5. Idempotent: skip when the npz exists and `validate()` passes
   (`--skip-existing` semantics).
