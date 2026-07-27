#!/usr/bin/env python3
import argparse
import copy
import glob
import json
import os
import subprocess
import sys
import time
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from tqdm import tqdm
import yaml


@dataclass
class Step:
    name: str
    commands: List[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_state(path: str, state: Dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=True)
    os.replace(tmp, path)


def _load_state(path: str) -> Dict:
    if not os.path.isfile(path):
        return {"created_at": _now_iso(), "steps": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock(lock_path: str, run_tag: str) -> None:
    if os.path.exists(lock_path):
        with open(lock_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        pid = int(info.get("pid", -1))
        if pid > 0 and _is_pid_alive(pid):
            raise RuntimeError(
                f"Run lock exists for tag={run_tag} (pid={pid}). "
                f"If this is stale, remove lock file: {lock_path}"
            )
        os.remove(lock_path)
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "run_tag": run_tag, "acquired_at": _now_iso()}, f, indent=2)


def _release_lock(lock_path: str) -> None:
    if os.path.exists(lock_path):
        os.remove(lock_path)


def _with_dataset_scope(base_dir: str, dataset: str, split_tag: str) -> str:
    norm = os.path.normpath(base_dir)
    parts = norm.split(os.sep)
    
    # Guard: if already a shared directory (no dataset suffix), skip scoping
    # This happens for all_5_datasets which reuses features from single runs
    if dataset == "all_5_datasets":
        return base_dir
    
    # Guard against legacy duplicated tails like ".../musiceval/musiceval".
    while len(parts) >= 2 and parts[-1] == dataset and parts[-2] == dataset:
        parts.pop()
    norm = os.sep.join(parts)
    tail_both = os.path.normpath(os.path.join(dataset, split_tag))
    if norm.endswith(tail_both):
        return norm
    if norm.endswith(os.path.normpath(dataset)):
        return os.path.join(norm, split_tag)
    return os.path.join(norm, dataset, split_tag)


def _write_runtime_scoped_config(
    base_cfg_path: str,
    state_dir: str,
    run_tag: str,
    project_root: str,
    dataset: str,
    split_tag: str,
) -> str:
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(base_cfg)

    out_cfg = cfg.get("outputs", {})
    outputs_root = out_cfg.get("root", "outputs")
    outputs_root = outputs_root if os.path.isabs(outputs_root) else os.path.join(project_root, outputs_root)
    checkpoints = _with_dataset_scope(
        out_cfg.get("checkpoints", os.path.join(outputs_root, "checkpoints")), dataset, split_tag
    )
    logs = _with_dataset_scope(out_cfg.get("logs", os.path.join(outputs_root, "logs")), dataset, split_tag)
    plots = _with_dataset_scope(out_cfg.get("plots", os.path.join(outputs_root, "plots")), dataset, split_tag)
    reports = _with_dataset_scope(out_cfg.get("reports", os.path.join(outputs_root, "reports")), dataset, split_tag)
    features_entropy = _with_dataset_scope(
        out_cfg.get("features_entropy", os.path.join(outputs_root, "features", "entropy")), dataset, split_tag
    )
    features_loss = _with_dataset_scope(
        out_cfg.get("features_loss", os.path.join(outputs_root, "features", "loss")), dataset, split_tag
    )
    for p in [checkpoints, logs, plots, reports, features_entropy, features_loss]:
        os.makedirs(p, exist_ok=True)

    cfg["outputs"] = {
        "root": outputs_root,
        "checkpoints": checkpoints,
        "logs": logs,
        "plots": plots,
        "reports": reports,
        "features_entropy": features_entropy,
        "features_loss": features_loss,
    }
    os.makedirs(os.path.join(logs, "tensorboard"), exist_ok=True)

    if "sae" in cfg:
        sae_out = cfg.get("sae", {}).get("output_dir", "features/sae")
        sae_out = sae_out if os.path.isabs(sae_out) else os.path.join(project_root, sae_out)
        sae_out = _with_dataset_scope(sae_out, dataset, split_tag)
        os.makedirs(sae_out, exist_ok=True)
        cfg["sae"]["output_dir"] = sae_out
    cfg.setdefault("data", {}).setdefault("feature_roots", {})["token_loss_root"] = features_loss

    cfg["project_root"] = project_root
    runtime_cfg_dir = os.path.join(state_dir, "runtime_configs")
    os.makedirs(runtime_cfg_dir, exist_ok=True)
    runtime_cfg_path = os.path.join(runtime_cfg_dir, f"{run_tag}.yaml")
    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return runtime_cfg_path


def run_command_with_tqdm(command: str, title: str, env: dict) -> float:
    t0 = time.time()
    with tqdm(desc=title, unit="s", dynamic_ncols=True) as pbar:
        proc = subprocess.Popen(command, shell=True, executable="/bin/bash", env=env)
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(1.0)
            pbar.update(1)
        if ret != 0:
            raise RuntimeError(f"Command failed ({ret}): {command}")
    return float(time.time() - t0)


def _count_csv_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # skip header
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _print_entropy_progress(cfg: Dict, project_root: str, split_group: str, split_name: str, entropy_dir: str) -> None:
    split_rel = cfg.get("data", {}).get("splits", {}).get(split_group, {}).get(split_name, "")
    split_csv = split_rel if os.path.isabs(split_rel) else os.path.join(project_root, split_rel)
    need = _count_csv_rows(split_csv)
    curve_dir = os.path.join(entropy_dir, split_name)
    have = len(glob.glob(os.path.join(curve_dir, "*_entropy.npy"))) if os.path.isdir(curve_dir) else 0
    missing = max(0, need - have)
    manifest = os.path.join(entropy_dir, f"entropy_manifest_{split_name}.csv")
    print(
        f"[entropy-progress] split={split_name} need={need} have={have} missing={missing} "
        f"curve_dir={curve_dir} manifest={manifest}"
    )


def _print_loss_progress(split_csv: str, split_name: str, loss_dir: str) -> None:
    need = _count_csv_rows(split_csv)
    curve_dir = os.path.join(loss_dir, split_name)
    have = len(glob.glob(os.path.join(curve_dir, "*_loss.csv"))) if os.path.isdir(curve_dir) else 0
    missing = max(0, need - have)
    manifest = os.path.join(loss_dir, f"loss_manifest_{split_name}.csv")
    print(
        f"[loss-progress] split={split_name} need={need} have={have} missing={missing} "
        f"curve_dir={curve_dir} manifest={manifest}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all 14 MusicEval experiments with per-step tqdm timing bars.")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Project root (default: env PROJECT_ROOT, else auto-detected from repo layout).")
    parser.add_argument("--config", type=str, default="config/paths.yaml")
    parser.add_argument("--torch-env", type=str, default="torch21")
    parser.add_argument("--musicdiscovery-env", type=str, default="musicdiscovery310")
    parser.add_argument("--audiobox-env", type=str, default="audiobox")
    parser.add_argument("--loss-feature-mode", type=str, default="per_codebook", choices=["per_codebook", "avg"])
    parser.add_argument("--hybrid-num-workers", type=int, default=8)
    parser.add_argument("--hybrid-prefetch-factor", type=int, default=8)
    parser.add_argument("--hybrid-cnn-batch-size", type=int, default=64)  # 4x for multi-GPU
    parser.add_argument("--hybrid-cnn-sae-only-batch-size", type=int, default=64)  # 2x for multi-GPU
    parser.add_argument("--dataset", type=str, default="musiceval")
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    parser.add_argument(
        "--extract-chunk-sec",
        type=float,
        default=0.0,
        help="Per-dataset: process each audio in non-overlapping windows of "
             "this many seconds (used by entropy/loss/SAE extractors).",
    )
    parser.add_argument(
        "--extract-pool-to-frames",
        type=int,
        default=0,
        help="Per-dataset: uniformly mean-pool concatenated per-clip features "
             "to this many frames along time (used with --extract-chunk-sec).",
    )
    parser.add_argument(
        "--extract-max-audio-sec",
        type=float,
        default=0.0,
        help="Per-dataset: cap each audio's duration in seconds (0 = unlimited).",
    )
    parser.add_argument(
        "--prepare-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run MusicEval split materialization in phase7_release before training.",
    )
    parser.add_argument("--skip-feature-extract", action="store_true")
    parser.add_argument(
        "--only-steps", type=str, default="",
        help="Comma-separated whitelist of step names to run (all others are skipped). "
             "E.g. --only-steps f03_sae_only_cnn",
    )
    parser.add_argument(
        "--skip-sae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip SAE feature extraction and all SAE-involved experiments.",
    )
    parser.add_argument("--with-aesthetics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from saved run state (command-level checkpoint).",
    )
    parser.add_argument("--reset-state", action="store_true", help="Delete previous run state for this run-tag.")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="State key for resume/lock. Default: musiceval14_<dataset>_<splits>.",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default="outputs/run_state",
        help="Directory storing run state and lock files.",
    )
    parser.add_argument(
        "--rerun-step",
        action="append",
        default=[],
        help="Step name to force rerun (can be repeated), e.g. --rerun-step f02_entropy_only_cnn",
    )
    parser.add_argument("--sae-world-size", type=int, default=1,
                        help="Number of GPUs for parallel SAE feature extraction.")
    parser.add_argument("--sae-gpu-ids", type=str, default="",
                        help="Comma-separated GPU IDs for SAE extraction (default: 0..world_size-1).")
    parser.add_argument("--sae-batch-size", type=int, default=16,
                        help="Audio files per GPU forward pass during SAE extraction.")
    return parser


def main() -> int:
    from lib.repro.data_paths import detect_project_root
    args = build_parser().parse_args()
    project_root = os.path.abspath(args.project_root) if args.project_root else detect_project_root()
    run_tag = args.run_tag or f"full14_{args.dataset}_{args.splits}"
    state_dir = args.state_dir if os.path.isabs(args.state_dir) else os.path.join(project_root, args.state_dir)
    os.makedirs(state_dir, exist_ok=True)
    state_file = os.path.join(state_dir, f"{run_tag}.state.json")
    lock_file = os.path.join(state_dir, f"{run_tag}.lock.json")

    if args.reset_state and os.path.isfile(state_file):
        os.remove(state_file)

    state = _load_state(state_file)
    sae_step_names = {
        "prep_sae_features",
        "f03_sae_only_cnn",
        "f05_entropy_sae_cnn",
        "f06_loss_sae_cnn",
        "f07_loss_entropy_sae_cnn",
    }
    state.update(
        {
            "run_tag": run_tag,
            "project_root": project_root,
            "splits": args.splits,
            "updated_at": _now_iso(),
        }
    )
    _save_state(state_file, state)
    _acquire_lock(lock_file, run_tag)

    base_cfg_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, args.config)
    release_cfg = _write_runtime_scoped_config(
        base_cfg_path=base_cfg_path,
        state_dir=state_dir,
        run_tag=run_tag,
        project_root=project_root,
        dataset=args.dataset,
        split_tag=args.splits,
    )
    with open(release_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    outputs_cfg = cfg.get("outputs", {})
    outputs_root = outputs_cfg.get("root", "outputs")
    outputs_root = outputs_root if os.path.isabs(outputs_root) else os.path.join(project_root, outputs_root)
    reports_root = outputs_cfg.get("reports", os.path.join(outputs_root, "reports"))
    reports_root = reports_root if os.path.isabs(reports_root) else os.path.join(project_root, reports_root)
    reports_dir = _with_dataset_scope(reports_root, args.dataset, args.splits)
    entropy_root = outputs_cfg.get("features_entropy", os.path.join(outputs_root, "features", "entropy"))
    entropy_root = entropy_root if os.path.isabs(entropy_root) else os.path.join(project_root, entropy_root)
    entropy_dir = _with_dataset_scope(entropy_root, args.dataset, args.splits)
    loss_root = outputs_cfg.get("features_loss", os.path.join(outputs_root, "features", "loss"))
    loss_root = loss_root if os.path.isabs(loss_root) else os.path.join(project_root, loss_root)
    loss_dir = _with_dataset_scope(loss_root, args.dataset, args.splits)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(entropy_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    # Use full_splits/{dataset} for each dataset (not musiceval splits)
    full_splits_dir = os.path.join(project_root, "data", "full_splits", args.dataset)
    source_split_paths: Dict[str, str] = {}
    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(full_splits_dir, f"{split_name}.csv")
        if not os.path.isfile(split_path):
            # Fallback to config splits if full_splits not available
            split_cfg = cfg.get("data", {}).get("splits", {}).get(args.splits, {})
            rel = split_cfg.get(split_name, "")
            split_path = rel if os.path.isabs(rel) else os.path.join(project_root, rel)
        source_split_paths[split_name] = split_path

    runtime_split_dir = os.path.join(state_dir, "runtime_splits", run_tag)
    os.makedirs(runtime_split_dir, exist_ok=True)
    runtime_split_paths = {k: os.path.join(runtime_split_dir, f"{k}.csv") for k in ["train", "val", "test"]}
    cfg.setdefault("data", {}).setdefault("splits", {}).setdefault(args.splits, {})
    for split_name in ["train", "val", "test"]:
        if args.skip_feature_extract:
            # Don't redirect to runtime_splits — they won't be created.
            # Resolve config split paths to absolute so training can find them.
            raw = cfg["data"]["splits"][args.splits].get(split_name, source_split_paths[split_name])
            cfg["data"]["splits"][args.splits][split_name] = (
                raw if os.path.isabs(raw) else os.path.join(project_root, raw)
            )
        else:
            cfg["data"]["splits"][args.splits][split_name] = runtime_split_paths[split_name]
    with open(release_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    entropy_args = (
        f'--entropy-manifest-csv "{entropy_dir}/entropy_manifest_train.csv" '
        f'--entropy-manifest-csv "{entropy_dir}/entropy_manifest_val.csv" '
        f'--entropy-manifest-csv "{entropy_dir}/entropy_manifest_test.csv"'
    )

    # Per-dataset extraction flags propagated to every feature extractor.
    extract_window_flags = ""
    if args.extract_chunk_sec > 0:
        extract_window_flags += f" --chunk-sec {args.extract_chunk_sec}"
    if args.extract_pool_to_frames > 0:
        extract_window_flags += f" --pool-to-frames {args.extract_pool_to_frames}"
    if args.extract_max_audio_sec > 0:
        extract_window_flags += f" --max-audio-sec {args.extract_max_audio_sec}"
    extract_window_flags = extract_window_flags.strip()

    steps: List[Step] = []
    if args.prepare_splits:
        steps.append(
            Step(
                "prep_data_splits",
                [
                    f'conda run -n "{args.torch_env}" python "scripts/data/preprocess_data.py" --config "{base_cfg_path}" musiceval-copy-splits'
                ],
            )
        )
    if not args.skip_feature_extract:
        steps.append(
            Step(
                "prep_loss_features",
                [
                    f'conda run -n "{args.torch_env}" python "scripts/features/extract_loss_curves.py" --config "{release_cfg}" --splits "{args.splits}" --split train --mode "{args.loss_feature_mode}" --source-split-csv "{source_split_paths["train"]}" --output-split-csv "{runtime_split_paths["train"]}" --manifest-csv "{loss_dir}/loss_manifest_train.csv" --out-dir "{loss_dir}" {extract_window_flags}',
                    f'conda run -n "{args.torch_env}" python "scripts/features/extract_loss_curves.py" --config "{release_cfg}" --splits "{args.splits}" --split val --mode "{args.loss_feature_mode}" --source-split-csv "{source_split_paths["val"]}" --output-split-csv "{runtime_split_paths["val"]}" --manifest-csv "{loss_dir}/loss_manifest_val.csv" --out-dir "{loss_dir}" {extract_window_flags}',
                    f'conda run -n "{args.torch_env}" python "scripts/features/extract_loss_curves.py" --config "{release_cfg}" --splits "{args.splits}" --split test --mode "{args.loss_feature_mode}" --source-split-csv "{source_split_paths["test"]}" --output-split-csv "{runtime_split_paths["test"]}" --manifest-csv "{loss_dir}/loss_manifest_test.csv" --out-dir "{loss_dir}" {extract_window_flags}',
                ],
            )
        )
        _sae_extra = f'--world-size {args.sae_world_size} --batch-size {args.sae_batch_size}'
        if args.sae_gpu_ids:
            _sae_extra += f' --gpu-ids "{args.sae_gpu_ids}"'
        steps.append(
            Step(
                "prep_sae_features",
                [
                    f'conda run -n "{args.musicdiscovery_env}" python "scripts/features/extract_sae_features.py" --config "{release_cfg}" --splits "{args.splits}" --split train {_sae_extra} {extract_window_flags}',
                    f'conda run -n "{args.musicdiscovery_env}" python "scripts/features/extract_sae_features.py" --config "{release_cfg}" --splits "{args.splits}" --split val {_sae_extra} {extract_window_flags}',
                    f'conda run -n "{args.musicdiscovery_env}" python "scripts/features/extract_sae_features.py" --config "{release_cfg}" --splits "{args.splits}" --split test {_sae_extra} {extract_window_flags}',
                ],
            )
        )
        steps.append(
            Step(
                "prep_entropy_features",
                [
                    f'conda run -n "{args.torch_env}" python "scripts/features/extract_entropy_curves.py" --config "{release_cfg}" --splits "{args.splits}" --split train --out-dir "{entropy_dir}" {extract_window_flags}',
                    f'conda run -n "{args.torch_env}" python "scripts/features/extract_entropy_curves.py" --config "{release_cfg}" --splits "{args.splits}" --split val --out-dir "{entropy_dir}" {extract_window_flags}',
                    f'conda run -n "{args.torch_env}" python "scripts/features/extract_entropy_curves.py" --config "{release_cfg}" --splits "{args.splits}" --split test --out-dir "{entropy_dir}" {extract_window_flags}',
                ],
            )
        )

    if args.with_aesthetics:
        steps.append(
            Step(
                "baseline_aesthetics",
                [
                    f'conda run -n "{args.audiobox_env}" python "scripts/baseline/aesthetics.py" --config "{release_cfg}" --splits "{args.splits}" --out-dir "{reports_dir}"'
                ],
            )
        )
        steps.append(
            Step(
                "baseline_mean_loss",
                [
                    f'conda run -n "{args.torch_env}" python "scripts/baseline/mean_loss.py" --config "{release_cfg}" --splits "{args.splits}" --out-dir "{reports_dir}"'
                ],
            )
        )
        steps.append(
            Step(
                "baseline_rescaled_eval",
                [
                    f'conda run -n "{args.torch_env}" python "scripts/eval/eval_rescaled.py" --config "{release_cfg}" --label_mode "{args.splits}" --out-dir "{reports_dir}"'
                ],
            )
        )

    steps.extend(
        [
            Step(
                "f01_loss_only_cnn",
                [
                    f'conda run -n "{args.torch_env}" python "training/loss_curve/train.py" --config "{release_cfg}" --splits "{args.splits}" --run-name "f01_loss_only_cnn_{args.splits}"',
                    f'conda run -n "{args.torch_env}" python "training/loss_curve/predict.py" --config "{release_cfg}" --splits "{args.splits}" --run-name "f01_loss_only_cnn_{args.splits}" --output-csv "{reports_dir}/f01_loss_only_cnn_{args.splits}_test_scores.csv"',
                ],
            ),
            Step(
                "f02_entropy_only_cnn",
                [
                    f'conda run -n "{args.torch_env}" python "training/entropy_curve/train.py" --config "{release_cfg}" --splits "{args.splits}" --run-name "f02_entropy_only_cnn_{args.splits}" {entropy_args}',
                    f'conda run -n "{args.torch_env}" python "training/entropy_curve/predict.py" --config "{release_cfg}" --splits "{args.splits}" --run-name "f02_entropy_only_cnn_{args.splits}" {entropy_args} --output-csv "{reports_dir}/f02_entropy_only_cnn_{args.splits}_test_scores.csv"',
                ],
            ),
            Step(
                "f03_sae_only_cnn",
                [
                    f'conda run -n "{args.torch_env}" python "training/hybrid/train_cnn.py" --config "{release_cfg}" --run_name "f03_sae_only_cnn_{args.splits}" --curve-mode none --batch_size {args.hybrid_cnn_sae_only_batch_size} --num-workers {args.hybrid_num_workers} --prefetch-factor {args.hybrid_prefetch_factor}',
                ],
            ),
            Step(
                "f04_loss_entropy_cnn",
                [
                    f'conda run -n "{args.torch_env}" python "training/loss_curve/train.py" --config "{release_cfg}" --splits "{args.splits}" --run-name "f04_loss_entropy_cnn_{args.splits}" {entropy_args}',
                    f'conda run -n "{args.torch_env}" python "training/loss_curve/predict.py" --config "{release_cfg}" --splits "{args.splits}" --run-name "f04_loss_entropy_cnn_{args.splits}" {entropy_args} --output-csv "{reports_dir}/f04_loss_entropy_cnn_{args.splits}_test_scores.csv"',
                ],
            ),
            Step(
                "f05_entropy_sae_cnn",
                [
                    f'conda run -n "{args.torch_env}" python "training/hybrid/train_cnn.py" --config "{release_cfg}" --run_name "f05_entropy_sae_cnn_{args.splits}" --curve-mode entropy --batch_size {args.hybrid_cnn_batch_size} --num-workers {args.hybrid_num_workers} --prefetch-factor {args.hybrid_prefetch_factor} {entropy_args}',
                ],
            ),
            Step(
                "f06_loss_sae_cnn",
                [
                    f'conda run -n "{args.torch_env}" python "training/hybrid/train_cnn.py" --config "{release_cfg}" --run_name "f06_loss_sae_cnn_{args.splits}" --curve-mode loss --batch_size {args.hybrid_cnn_batch_size} --num-workers {args.hybrid_num_workers} --prefetch-factor {args.hybrid_prefetch_factor}',
                ],
            ),
            Step(
                "f07_loss_entropy_sae_cnn",
                [
                    f'conda run -n "{args.torch_env}" python "training/hybrid/train_cnn.py" --config "{release_cfg}" --run_name "f07_loss_entropy_sae_cnn_{args.splits}" --curve-mode loss_entropy --batch_size {args.hybrid_cnn_batch_size} --num-workers {args.hybrid_num_workers} --prefetch-factor {args.hybrid_prefetch_factor} {entropy_args}',
                ],
            ),
            Step(
                "eval_14_experiments",
                [
                    f'conda run -n "{args.torch_env}" python "scripts/eval/eval_14_experiments.py" --reports-dir "{reports_dir}" --split-tag "{args.splits}"'
                ],
            ),
        ]
    )
    if args.skip_sae and args.resume:
        changed = 0
        for name in sae_step_names:
            st = state.setdefault("steps", {}).get(name)
            if not st:
                continue
            if st.get("status") != "completed":
                st["status"] = "skipped"
                st["skipped_at"] = _now_iso()
                st["skip_reason"] = "skip_sae_enabled"
                changed += 1
        if changed > 0:
            state["updated_at"] = _now_iso()
            _save_state(state_file, state)
            print(f"[skip-sae] marked {changed} SAE-related step states as skipped in resume state.")

    if args.skip_sae:
        before = len(steps)
        steps = [s for s in steps if s.name not in sae_step_names]
        print(f"[skip-sae] removed {before - len(steps)} SAE-related steps.")

    if args.only_steps:
        whitelist = {s.strip() for s in args.only_steps.split(",") if s.strip()}
        before = len(steps)
        steps = [s for s in steps if s.name in whitelist]
        print(f"[only-steps] keeping {len(steps)}/{before} steps: {sorted(whitelist)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f'{project_root}:{env.get("PYTHONPATH", "")}'
    # Ensure conda is on PATH for subprocess shells that invoke `conda run`.
    _conda_bin = os.path.expanduser("~/miniconda3/bin")
    if os.path.isdir(_conda_bin) and _conda_bin not in env.get("PATH", ""):
        env["PATH"] = f'{_conda_bin}:{env.get("PATH", "")}'
    os.chdir(project_root)

    try:
        forced_steps = set(args.rerun_step or [])
        if forced_steps:
            print(f"[rerun-step] forcing rerun for: {sorted(forced_steps)}")
        total_steps = len(steps)
        for step in steps:
            step_state = state.setdefault("steps", {}).setdefault(step.name, {})
            if step.name in forced_steps:
                step_state.pop("completed_commands", None)
                step_state.pop("completed_at", None)
                step_state["status"] = "pending"
                step_state["forced_rerun_at"] = _now_iso()
                _save_state(state_file, state)

        completed_steps_before = 0
        if args.resume:
            for step in steps:
                st = state.setdefault("steps", {}).setdefault(step.name, {})
                if st.get("status") == "completed":
                    completed_steps_before += 1
        timeline_rows = []
        timeline_dir = os.path.join(state_dir, "timelines")
        os.makedirs(timeline_dir, exist_ok=True)
        timeline_csv = os.path.join(timeline_dir, f"{run_tag}.csv")

        with tqdm(
            total=total_steps,
            initial=completed_steps_before,
            desc="overall",
            unit="step",
            dynamic_ncols=True,
        ) as overall_pbar:
            for idx, step in enumerate(steps, start=1):
                step_state = state.setdefault("steps", {}).setdefault(step.name, {})
                overall_pbar.set_postfix_str(f"{idx}/{total_steps}:{step.name}", refresh=False)
                if args.resume and step_state.get("status") == "completed":
                    print(f"\n=== [{idx}/{total_steps}] {step.name} (skip: already completed) ===")
                    continue

                step_state.update({"status": "running", "started_at": _now_iso()})
                completed_cmds = set(step_state.get("completed_commands", []))
                _save_state(state_file, state)
                step_t0 = time.time()

                print(f"\n=== [{idx}/{total_steps}] {step.name} ===")
                for cmd_idx, cmd in enumerate(step.commands, start=1):
                    if args.resume and cmd_idx in completed_cmds:
                        print(f"[resume] skip command {cmd_idx}/{len(step.commands)} for {step.name}")
                        continue
                    if step.name == "prep_entropy_features":
                        if "--split train" in cmd:
                            _print_entropy_progress(cfg, project_root, args.splits, "train", entropy_dir)
                        elif "--split val" in cmd:
                            _print_entropy_progress(cfg, project_root, args.splits, "val", entropy_dir)
                        elif "--split test" in cmd:
                            _print_entropy_progress(cfg, project_root, args.splits, "test", entropy_dir)
                    if step.name == "prep_loss_features":
                        if "--split train" in cmd:
                            _print_loss_progress(source_split_paths["train"], "train", loss_dir)
                        elif "--split val" in cmd:
                            _print_loss_progress(source_split_paths["val"], "val", loss_dir)
                        elif "--split test" in cmd:
                            _print_loss_progress(source_split_paths["test"], "test", loss_dir)
                    title = f"{step.name} ({cmd_idx}/{len(step.commands)})"
                    elapsed_s = run_command_with_tqdm(cmd, title=title, env=env)
                    timeline_rows.append(
                        {
                            "step_idx": idx,
                            "step_name": step.name,
                            "command_idx": cmd_idx,
                            "command_total": len(step.commands),
                            "elapsed_sec": round(elapsed_s, 3),
                        }
                    )
                    print(f"[timeline] {step.name} cmd {cmd_idx}/{len(step.commands)} took {elapsed_s:.1f}s")
                    completed_cmds.add(cmd_idx)
                    step_state["completed_commands"] = sorted(completed_cmds)
                    step_state["updated_at"] = _now_iso()
                    _save_state(state_file, state)

                step_elapsed = float(time.time() - step_t0)
                step_state.update({"status": "completed", "completed_at": _now_iso(), "elapsed_sec": round(step_elapsed, 3)})
                _save_state(state_file, state)
                overall_pbar.update(1)
                print(f"[timeline] step {idx}/{total_steps} {step.name} finished in {step_elapsed:.1f}s")
        if timeline_rows:
            with open(timeline_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step_idx", "step_name", "command_idx", "command_total", "elapsed_sec"]
                )
                writer.writeheader()
                for row in timeline_rows:
                    writer.writerow(row)
            print(f"[timeline] wrote {timeline_csv}")
    except Exception as e:
        state.setdefault("meta", {})["last_error"] = str(e)
        state["updated_at"] = _now_iso()
        _save_state(state_file, state)
        raise
    finally:
        _release_lock(lock_file)

    print("\nAll requested MusicEval experiments completed.")
    state.setdefault("meta", {})["finished_at"] = _now_iso()
    _save_state(state_file, state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        raise SystemExit(1)
