#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import yaml


@dataclass
class Job:
    name: str
    scope: str
    config_path: str
    log_path: str
    extract_flags: Dict[str, float]
    gpus_per_job: int = 1
    only_steps: str = ""


@dataclass
class RunningJob:
    job: Job
    gpu_ids: List[int]
    process: subprocess.Popen
    started_at: float

    @property
    def gpu_id(self) -> int:
        return self.gpu_ids[0]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def _query_gpus() -> List[Dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    rows: List[Dict[str, int]] = []
    for line in out.splitlines():
        idx, mem_total, mem_used, util = [x.strip() for x in line.split(",")]
        rows.append(
            {
                "index": int(idx),
                "memory_total_mb": int(mem_total),
                "memory_used_mb": int(mem_used),
                "memory_free_mb": int(mem_total) - int(mem_used),
                "util_pct": int(util),
            }
        )
    return rows


def _eligible_gpu_ids(
    all_rows: List[Dict[str, int]],
    allowed_gpu_ids: List[int],
    min_free_memory_mb: int,
    max_util_pct: int,
) -> List[int]:
    allowed = set(allowed_gpu_ids) if allowed_gpu_ids else None
    out = []
    for row in all_rows:
        idx = row["index"]
        if allowed is not None and idx not in allowed:
            continue
        if row["memory_free_mb"] < min_free_memory_mb:
            continue
        if row["util_pct"] > max_util_pct:
            continue
        out.append(idx)
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run full-scale 14-experiment matrix in parallel across GPUs.")
    p.add_argument("--project-root", type=str, default=None,
                   help="Project root (default: env PROJECT_ROOT, else auto-detected from repo layout).")
    p.add_argument("--base-config", type=str, default="config/paths.yaml")
    p.add_argument("--full-config", type=str, default="")
    p.add_argument("--torch-env", type=str, default="torch21")
    p.add_argument("--musicdiscovery-env", type=str, default="musicdiscovery310")
    p.add_argument("--audiobox-env", type=str, default="audiobox")
    p.add_argument("--splits", type=str, default="clean", choices=["clean"])
    p.add_argument("--dry-run", action="store_true")
    return p


def _build_jobs(project_root: str, base_cfg_path: str, full_cfg_path: str, split_tag: str) -> List[Job]:
    base_cfg = _load_yaml(base_cfg_path)
    full_cfg = _load_yaml(full_cfg_path)

    include_scopes = full_cfg.get("execution", {}).get("include_scopes", ["large_scale_single", "large_scale_merged"])
    jobs: List[Job] = []

    generated_cfg_root = os.path.join(project_root, "outputs", "run_state", "full_generated_configs")
    logs_root = os.path.join(project_root, "outputs", "run_state", "full_parallel_logs")
    os.makedirs(generated_cfg_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    for scope in include_scopes:
        entries = full_cfg.get("datasets", {}).get(scope, [])
        for entry in entries:
            if not bool(entry.get("enabled", True)):
                continue
            name = str(entry["name"])
            split_cfg = entry.get("splits", {}).get(split_tag, {})
            required_keys = {"train", "val", "test"}
            if not required_keys.issubset(split_cfg.keys()):
                raise ValueError(f"Dataset '{name}' in scope '{scope}' missing split keys for '{split_tag}'")

            merged = copy.deepcopy(base_cfg)
            merged["data"]["splits"][split_tag] = {
                "train": split_cfg["train"],
                "val": split_cfg["val"],
                "test": split_cfg["test"],
            }

            out_root_rel = os.path.join("outputs", "full")
            feature_root_rel = os.path.join("features")
            merged["outputs"] = {
                "root": out_root_rel,
                "checkpoints": os.path.join(out_root_rel, "checkpoints", name, split_tag),
                "logs": os.path.join(out_root_rel, "logs", name, split_tag),
                "plots": os.path.join(out_root_rel, "plots", name, split_tag),
                "reports": os.path.join(out_root_rel, "reports", name, split_tag),
                "features_entropy": os.path.join(feature_root_rel, "entropy", name, split_tag),
                "features_loss": os.path.join(feature_root_rel, "loss", name, split_tag),
            }
            merged.setdefault("data", {}).setdefault("feature_roots", {})["token_loss_root"] = os.path.join(
                feature_root_rel, "loss", name, split_tag
            )
            if "sae" in merged:
                merged["sae"]["output_dir"] = os.path.join(feature_root_rel, "sae", name, split_tag)

            # all_5_datasets reads features from per-dataset subdirs under the shared feature root
            if name == "all_5_datasets":
                merged["outputs"]["features_entropy"] = os.path.join(feature_root_rel, "entropy")
                merged["outputs"]["features_loss"] = os.path.join(feature_root_rel, "loss")
                merged.setdefault("data", {}).setdefault("feature_roots", {})["token_loss_root"] = os.path.join(
                    feature_root_rel, "loss"
                )
                if "sae" in merged:
                    merged["sae"]["output_dir"] = os.path.join(feature_root_rel, "sae")

            cfg_path = os.path.join(generated_cfg_root, f"{scope}_{name}_{split_tag}.yaml")
            _save_yaml(cfg_path, merged)
            log_path = os.path.join(logs_root, f"{scope}_{name}_{split_tag}.log")
            ef = entry.get("extract_flags", {}) or {}
            extract_flags: Dict[str, float] = {
                "chunk_sec": float(ef.get("chunk_sec", 0.0) or 0.0),
                "pool_to_frames": int(ef.get("pool_to_frames", 0) or 0),
                "max_audio_sec": float(ef.get("max_audio_sec", 0.0) or 0.0),
            }
            gpus_per_job = int(entry.get("gpus_per_job", 1) or 1)
            only_steps = str(entry.get("only_steps", "") or "")
            jobs.append(Job(
                name=name, scope=scope, config_path=cfg_path, log_path=log_path,
                extract_flags=extract_flags, gpus_per_job=gpus_per_job, only_steps=only_steps,
            ))
    return jobs


def main() -> int:
    from lib.repro.data_paths import detect_project_root
    args = build_parser().parse_args()
    project_root = os.path.abspath(args.project_root) if args.project_root else detect_project_root()
    base_cfg = args.base_config if os.path.isabs(args.base_config) else os.path.join(project_root, args.base_config)
    if args.full_config:
        full_cfg = args.full_config if os.path.isabs(args.full_config) else os.path.join(project_root, args.full_config)
    else:
        base_cfg_obj = _load_yaml(base_cfg)
        rel = base_cfg_obj.get("data", {}).get("full_scale", {}).get("dataset_config", "config/data/full_datasets.yaml")
        full_cfg = rel if os.path.isabs(rel) else os.path.join(project_root, rel)

    full_cfg_obj = _load_yaml(full_cfg)
    parallel_cfg = full_cfg_obj.get("gpu_parallel", {})
    if not parallel_cfg.get("enabled", True):
        raise RuntimeError(f"gpu_parallel.enabled=false in {full_cfg}; enable it for full-scale parallel run.")

    jobs = _build_jobs(project_root, base_cfg, full_cfg, split_tag=args.splits)
    if not jobs:
        print("No enabled jobs found in full config.")
        return 0

    detect_cfg = parallel_cfg.get("detection", {})
    min_free = int(detect_cfg.get("min_free_memory_mb", 60000))
    max_util = int(detect_cfg.get("max_utilization_pct", 35))
    poll_interval = int(detect_cfg.get("probe_interval_sec", 20))
    max_jobs = int(parallel_cfg.get("max_concurrent_jobs", 8))
    gpu_ids = [int(x) for x in parallel_cfg.get("gpu_ids", [])]

    gpu_rows = _query_gpus()
    eligible_start = _eligible_gpu_ids(gpu_rows, gpu_ids, min_free, max_util)
    if not eligible_start:
        raise RuntimeError("No eligible GPUs found. Check nvidia-smi state or relax thresholds in full_datasets.yaml")
    print(f"[gpu-detect] eligible GPUs at startup: {eligible_start}")
    if args.dry_run:
        for job in jobs:
            print(f"[dry-run] {job.scope}/{job.name} config={job.config_path}")
        return 0

    pending = list(jobs)
    running: List[RunningJob] = []
    results: List[Dict[str, str]] = []
    os.chdir(project_root)
    print(f"[gpu-detect] pending jobs: {len(pending)}, max concurrent: {max_jobs}")

    while pending or running:
        still_running: List[RunningJob] = []
        for rj in running:
            code = rj.process.poll()
            if code is None:
                still_running.append(rj)
                continue
            elapsed = int(time.time() - rj.started_at)
            gpu_str = ",".join(str(g) for g in rj.gpu_ids)
            results.append(
                {
                    "scope": rj.job.scope,
                    "dataset": rj.job.name,
                    "gpu_id": gpu_str,
                    "exit_code": str(code),
                    "elapsed_sec": str(elapsed),
                    "log_path": rj.job.log_path,
                    "finished_at": _now_iso(),
                }
            )
            status = "ok" if code == 0 else "failed"
            print(f"[finish] {rj.job.scope}/{rj.job.name} on gpu(s) {gpu_str} -> {status} (exit={code}, {elapsed}s)")
        running = still_running

        if pending and len(running) < max_jobs:
            gpu_rows = _query_gpus()
            eligible = _eligible_gpu_ids(gpu_rows, gpu_ids, min_free, max_util)
            busy = {g for rj in running for g in rj.gpu_ids}
            free = [g for g in eligible if g not in busy]
            while pending and len(running) < max_jobs:
                job = pending[0]
                if len(free) < job.gpus_per_job:
                    break
                pending.pop(0)
                assigned = [free.pop(0) for _ in range(job.gpus_per_job)]
                run_tag = f"full14_{job.name}_{args.splits}"
                conda_bin = shutil.which("conda") or os.path.expanduser("~/miniconda3/bin/conda")
                cmd = [
                    conda_bin,
                    "run",
                    "-n",
                    args.torch_env,
                    "python",
                    "scripts/run/run_musiceval_14_experiments.py",
                    "--project-root",
                    project_root,
                    "--config",
                    job.config_path,
                    "--torch-env",
                    args.torch_env,
                    "--musicdiscovery-env",
                    args.musicdiscovery_env,
                    "--audiobox-env",
                    args.audiobox_env,
                    "--dataset",
                    job.name,
                    "--splits",
                    args.splits,
                    "--run-tag",
                    run_tag,
                    "--no-prepare-splits",
                ]
                if not full_cfg_obj.get("execution", {}).get("with_aesthetics", True):
                    cmd.append("--no-with-aesthetics")
                if full_cfg_obj.get("execution", {}).get("skip_feature_extract", False):
                    cmd.append("--skip-feature-extract")
                if job.only_steps:
                    cmd += ["--only-steps", job.only_steps]
                # Pass-through per-dataset extraction window/chunk/pool flags.
                ef = job.extract_flags
                if ef.get("chunk_sec", 0.0) > 0:
                    cmd += ["--extract-chunk-sec", str(ef["chunk_sec"])]
                if int(ef.get("pool_to_frames", 0)) > 0:
                    cmd += ["--extract-pool-to-frames", str(int(ef["pool_to_frames"]))]
                if ef.get("max_audio_sec", 0.0) > 0:
                    cmd += ["--extract-max-audio-sec", str(ef["max_audio_sec"])]

                env = os.environ.copy()
                gpu_str = ",".join(str(g) for g in assigned)
                env["CUDA_VISIBLE_DEVICES"] = gpu_str
                lf = open(job.log_path, "a", encoding="utf-8")
                lf.write(f"\n[{_now_iso()}] launch on gpu(s)={gpu_str}: {' '.join(cmd)}\n")
                lf.flush()
                proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf, cwd=project_root)
                running.append(RunningJob(job=job, gpu_ids=assigned, process=proc, started_at=time.time()))
                print(f"[launch] {job.scope}/{job.name} on gpu(s) {gpu_str} (pid={proc.pid})")

        if pending or running:
            time.sleep(poll_interval)

    summary_path = os.path.join(project_root, "outputs", "run_state", f"full_parallel_summary_{args.splits}.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scope", "dataset", "gpu_id", "exit_code", "elapsed_sec", "log_path", "finished_at"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[summary] wrote {summary_path}")

    failed = [r for r in results if r["exit_code"] != "0"]
    if failed:
        print(f"[result] {len(failed)} jobs failed. See logs in outputs/run_state/full_parallel_logs")
        return 1
    print("[result] all full-scale jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
