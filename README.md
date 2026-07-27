# MEva — Automatic Music Evaluation with Model-Intrinsic Signals

Code and demo for the ISMIR 2026 paper
**“Do Music Generative Models Understand Musicality? Automatic Music Evaluation with Model-Intrinsic Signals”**
(Xiaosha Li, Chun Liu, Ziyu Wang).

- 📄 Paper: [`docs/assets/paper.pdf`](docs/assets/paper.pdf)
- 🔊 Demo page: <https://yoev.github.io/MEva/>
- 🤗 Trained evaluator checkpoints: <https://huggingface.co/Dev4IC>

MEva evaluates generated music **without an external audio verifier**: it reads three
intrinsic signals of a single frozen MusicGen under teacher forcing — per-token **loss**
ℓ_t, next-token **predictive entropy** H_t, and **sparse-autoencoder (SAE) latents** of the
hidden states — and maps them to human ratings with a lightweight CNN. Evaluated on five
human-rated benchmarks (MusicEval, SongEval, AIME, MusicPref, Music Arena).

## Repository layout

```
lib/            shared utilities (Elo rescoring, audio windowing, delay-pattern forward, nets)
scripts/
  features/     loss / entropy / SAE feature extraction (delay pattern by default)
  data/         benchmark ingestion, split generation, Elo target fitting
  baseline/     Audiobox Aesthetics + mean-loss baselines
  eval/         correlation evaluation over the 14 experiment variants
  run/          end-to-end pipelines and environment setup
training/       CNN evaluator training (single-signal and hybrid variants)
config/         data / model / training configs
docs/           demo page (GitHub Pages)
```

## Setup

Python ≥ 3.10, PyTorch 2.1, `transformers<=4.44` (MusicGen teacher forcing),
`audiocraft`-compatible EnCodec. SAE checkpoints come from
[PapayaResearch/musicdiscovery](https://github.com/PapayaResearch/musicdiscovery)
(Singh et al.) — `scripts/run/setup_sae_musicdiscovery.sh` downloads them.

## Feature extraction (delay pattern)

MusicGen is trained with a codebook **delay pattern**; teacher-forcing must interleave
codebooks the same way (`lib/repro/musicgen_pattern.py`). Extraction defaults to the
paper protocol:

```bash
python scripts/features/extract_loss_curves.py    --pattern delay ...
python scripts/features/extract_entropy_curves.py --pattern delay ...
python scripts/features/extract_sae_features.py   ...
```

(`--pattern parallel` reproduces the legacy non-delayed variant.)

## Training & evaluation

```bash
python training/hybrid/train_cnn.py --config config/model/experiments/f07_loss_entropy_sae/cnn.yaml
python scripts/eval/eval_14_experiments.py --reports-dir outputs/reports
```

The 14 experiment variants (7 signal combinations × MusicGen-small/large) and their
per-benchmark / pooled checkpoints are released on Hugging Face.

## Citation

```bibtex
@inproceedings{li2026meva,
  title     = {Do Music Generative Models Understand Musicality? Automatic Music Evaluation with Model-Intrinsic Signals},
  author    = {Li, Xiaosha and Liu, Chun and Wang, Ziyu},
  booktitle = {Proc. of the 27th Int. Society for Music Information Retrieval Conf. (ISMIR)},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). MusicGen weights (Meta,
CC-BY-NC 4.0), SAE checkpoints (Singh et al.), and benchmark audio remain under their
original licenses; demo audio consists of 2-second research excerpts from the
MusicEval benchmark.
