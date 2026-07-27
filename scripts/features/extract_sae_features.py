"""Launcher that forwards flags to ``extract_sae_features_musicdiscovery.py``.

Adds transparent pass-through for the rater-aligned / chunked / pooled
window flags introduced for AIME, Music Arena and SongEval.  When these
flags are left at their defaults the behaviour is identical to the
pre-existing MusicEval extraction path.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, SCRIPTS_DIR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(SCRIPTS_DIR), "config", "paths.yaml"),
    )
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--mode", type=str, default="sharded", choices=["sharded", "monolithic"])
    parser.add_argument("--max-seq-len", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--world-size", type=int, default=1,
                        help="Number of parallel GPU processes for extraction.")
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="Comma-separated GPU IDs to use (e.g. '0,1,2,3'). "
                             "Defaults to 0..world_size-1.")
    parser.add_argument("--chunk-sec", type=float, default=0.0,
                        help="If > 0, each audio is processed in non-overlapping "
                             "windows of this many seconds (needed for long songs).")
    parser.add_argument("--pool-to-frames", type=int, default=0,
                        help="Uniformly mean-pool concatenated SAE features to this "
                             "many frames along time (0 = no pooling).")
    parser.add_argument("--max-audio-sec", type=float, default=0.0,
                        help="Cap audio duration in seconds (0 = unlimited).")
    known = parser.parse_args()

    with open(known.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "project_root" not in cfg:
        env_root = os.environ.get("PROJECT_ROOT", "")
        if not env_root:
            env_root = os.path.abspath(os.path.join(os.path.dirname(known.config), "../.."))
        cfg["project_root"] = env_root
    project_root = cfg["project_root"]

    sae_cfg = cfg.get("sae", {})
    backend = sae_cfg.get("backend", "musicdiscovery")
    if backend != "musicdiscovery":
        raise ValueError("Only musicdiscovery backend is supported in phase7_release.")

    md_script = os.path.join(THIS_DIR, "extract_sae_features_musicdiscovery.py")
    checkpoint_dir = sae_cfg.get("musicdiscovery_checkpoint_dir", "")
    if not checkpoint_dir:
        raise ValueError("Missing sae.musicdiscovery_checkpoint_dir in config/paths.yaml")
    checkpoint_dir = (checkpoint_dir if os.path.isabs(checkpoint_dir)
                      else os.path.join(project_root, checkpoint_dir))
    required = ["cfg.json", "sae_weights.safetensors", "sparsity.safetensors"]
    missing = [f for f in required if not os.path.isfile(os.path.join(checkpoint_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing SAE checkpoint files in {checkpoint_dir}: {missing}. "
            "Run: bash phase7_release/scripts/run/setup_sae_musicdiscovery.sh"
        )
    output_dir = sae_cfg.get("output_dir", features/sae")
    output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(project_root, output_dir)

    if known.gpu_ids:
        gpu_ids = [g.strip() for g in known.gpu_ids.split(",") if g.strip()]
    else:
        gpu_ids = [str(i) for i in range(known.world_size)]
    if len(gpu_ids) < known.world_size:
        raise ValueError(f"--gpu-ids has {len(gpu_ids)} IDs but --world-size={known.world_size}")

    env_base = os.environ.copy()
    conda_lib = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "lib"))
    env_base["LD_LIBRARY_PATH"] = conda_lib + ":" + env_base.get("LD_LIBRARY_PATH", "")

    base_cmd = [
        "python", md_script,
        "--project-root", project_root,
        "--config", known.config,
        "--split", known.split,
        "--splits", known.splits,
        "--checkpoint-dir", checkpoint_dir,
        "--output-dir", output_dir,
        "--mode", known.mode,
        "--max-seq-len", str(known.max_seq_len),
        "--batch-size", str(known.batch_size),
        "--world-size", str(known.world_size),
        "--chunk-sec", str(known.chunk_sec),
        "--pool-to-frames", str(known.pool_to_frames),
        "--max-audio-sec", str(known.max_audio_sec),
    ]

    if known.world_size == 1:
        env = dict(env_base)
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        result = subprocess.run(base_cmd + ["--rank", "0"], check=False, env=env)
        return int(result.returncode)

    procs = []
    for rank in range(known.world_size):
        env = dict(env_base)
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
        cmd = base_cmd + ["--rank", str(rank)]
        print(f"[spawn] rank={rank} GPU={gpu_ids[rank]} {' '.join(cmd[-4:])}")
        procs.append(subprocess.Popen(cmd, env=env))

    failed = []
    for rank, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failed.append(rank)
    if failed:
        print(f"[error] ranks failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
