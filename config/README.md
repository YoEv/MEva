# Phase7 Release — Config Layout

All paths below are under `config/`.

## Directory map

```text
paths.yaml                # runtime paths used by all active scripts
                         # includes training.defaults (epoch / early-stop / lr-decay)

data/
  score_scaling.yaml      # BT vs MSE pairwise scaling, label column names

model/
  experiments/
    registry.yaml           # 7 families × 1 backbone (cnn) → model + training file paths
    f01_loss_only/          # … f07_loss_entropy_sae/
      cnn.yaml
  sae_sparse_autoencoder.yaml   # train SAE on MusicGen activations (prerequisite for SAE features)
  sae_verifier.yaml             # optional verifier on precomputed SAE vectors

training/
  experiments/
    f01_loss_only/ … f07_loss_entropy_sae/
      cnn.yaml                 # optimizer, manifest requirements
  sae_sparse_autoencoder.yaml
  sae_verifier.yaml

analysis/
  analysis.yaml           # segment scoring, curve alignment, SAE interpretability
  baseline_eval.yaml      # quantile / linear regression baselines (exp12-style)
```

## Experiment matrix (fixed)

Per dataset track (e.g. MusicEval smoke, or each of 5 DBs, or merged 5-DB):

- **7 families** × **1 backbone (CNN)** = **7** training runs. Transformer backbones were removed.
- Split tag is fixed to `clean`; noisy was removed.
- Index: `model/experiments/registry.yaml`.

## Defaults for wrappers

Release Python entrypoints default to:

- `--config` default: `config/paths.yaml`
