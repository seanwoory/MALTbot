#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]


def kst_today() -> str:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d")


def first_numeric(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for v in x.values():
            n = first_numeric(v)
            if n is not None:
                return n
    if isinstance(x, list):
        for v in x:
            n = first_numeric(v)
            if n is not None:
                return n
    return None


def collect_numerics(x: Any, out: list[float]) -> None:
    if isinstance(x, (int, float)):
        out.append(float(x))
        return
    if isinstance(x, dict):
        for v in x.values():
            collect_numerics(v, out)
        return
    if isinstance(x, list):
        for v in x:
            collect_numerics(v, out)


def metric_unit_for_task(task: str) -> str | None:
    if task == "matbench_mp_e_form":
        return "eV/atom"
    if task == "matbench_mp_gap":
        return "eV"
    return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def gpu_info() -> tuple[bool | None, str | None, str | None]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_type = torch.cuda.get_device_name(0) if cuda_available else None
        return cuda_available, gpu_type, torch.__version__
    except Exception:
        return None, None, None


def load_exp(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.exp:
        p = Path(args.exp)
    else:
        p = REPO / "configs" / "experiments" / f"{args.exp_name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Experiment config not found: {p}")
    exp = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return exp, p


def append_results_md(line: str) -> None:
    f = REPO / "RESULTS.md"
    with f.open("a", encoding="utf-8") as fp:
        fp.write("\n" + line)


def base_payload(exp_name: str, task: str, seed: int, model_config: str, note: str, exp_path: Path, out_file: Path) -> dict[str, Any]:
    cuda_available, gpu_type, torch_ver = gpu_info()
    payload: dict[str, Any] = {
        "experiment": exp_name,
        "task": {"name": task},
        "status": "success",
        "metric": None,
        "metric_name": "MAE",
        "metric_value": None,
        "metric_unit": metric_unit_for_task(task),
        "fold_scores": None,
        "mean": None,
        "std": None,
        "train_wall_time_sec": None,
        "dataset_size": None,
        "num_params": None,
        "seed": seed,
        "model_config": model_config,
        "note": note,
        "exp_config_path": str(exp_path.relative_to(REPO)),
        "output_path": str(out_file.relative_to(REPO)),
        "env": {
            "python": sys.version,
            "git_commit": git_commit(),
            "torch": torch_ver,
            "cuda_available": cuda_available,
            "gpu_type": gpu_type,
        },
    }
    return payload


def enrich_from_task_scores(payload: dict[str, Any], scores: Any) -> None:
    payload.setdefault("task", {})["scores"] = scores

    if isinstance(scores, dict):
        # Prefer explicit fold containers if present.
        for key in ("fold_scores", "folds", "scores_by_fold"):
            if key in scores and isinstance(scores[key], (dict, list)):
                payload["fold_scores"] = scores[key]
                break
        if payload.get("fold_scores") is None:
            # Best-effort: treat top-level dict as fold-score map if values are numeric or nested numeric.
            if scores:
                payload["fold_scores"] = scores

    n = first_numeric(scores)
    vals: list[float] = []
    if payload.get("fold_scores") is not None:
        collect_numerics(payload["fold_scores"], vals)
    else:
        collect_numerics(scores, vals)

    if n is not None:
        payload["metric"] = n
        payload["metric_value"] = n
    else:
        payload["metric"] = "RECORDED"
        payload["metric_value"] = None

    if vals:
        payload["mean"] = statistics.fmean(vals)
        payload["std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0


def finish_and_log(payload: dict[str, Any], date: str, task: str, model_config: str, note: str, out_file: Path) -> None:
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    metric_for_line = payload.get("metric")
    if isinstance(metric_for_line, float):
        metric_for_line = f"{metric_for_line:.6f}"

    line = f"{date} | {task} | {model_config} | METRIC={metric_for_line} | {note} | {out_file.relative_to(REPO)}"
    append_results_md(line)
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp")
    g.add_argument("--exp-name")
    args = ap.parse_args()

    exp, exp_path = load_exp(args)
    exp_name = exp.get("name") or (Path(args.exp).stem if args.exp else args.exp_name)
    task = exp.get("task", "matbench_mp_e_form")
    seed = int(exp.get("seed", 42))
    model_config = exp.get("model_config", exp_name)
    note = exp.get("note", "")
    enabled = bool(exp.get("enabled", True))
    runner = exp.get("runner", "chgnet")

    date = os.getenv("MALTBOT_DATE", kst_today())
    batch = os.getenv("MALTBOT_BATCH_RUN_NAME", "batch")

    out_dir = REPO / "results" / "daily" / date / batch / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"

    payload = base_payload(exp_name, task, seed, model_config, note, exp_path, out_file)

    if not enabled:
        payload["status"] = "skipped"
        payload["metric"] = "SKIPPED"
        payload["error_message"] = "Experiment disabled in YAML (enabled: false)."
        finish_and_log(payload, date, task, model_config, note, out_file)
        return

    if runner != "chgnet":
        payload["status"] = "skipped"
        payload["metric"] = "SKIPPED"
        payload["error_message"] = f"Runner '{runner}' not implemented yet."
        finish_and_log(payload, date, task, model_config, note, out_file)
        return

    base_cfg = REPO / "configs" / "chgnet_mp_e_form.yaml"
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    cfg["seed"] = seed
    cfg["run_name"] = exp_name
    cfg["output_root"] = str((REPO / "results" / "daily" / date / batch).relative_to(REPO))
    cfg.setdefault("task", {})["name"] = task

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(cfg, tf, sort_keys=False)
        temp_cfg = tf.name

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/run_chgnet_mp_e_form.py", "--config", temp_cfg],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        payload["train_wall_time_sec"] = round(time.perf_counter() - t0, 3)
        payload["run"] = {
            "returncode": proc.returncode,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-30:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-30:]),
        }
        if proc.returncode != 0:
            payload["status"] = "error"
            payload["metric"] = "ERROR"
            payload["error_message"] = f"runner exited with code {proc.returncode}"
            payload["traceback_or_stderr_tail"] = payload["run"]["stderr_tail"]
    except Exception as e:
        payload["train_wall_time_sec"] = round(time.perf_counter() - t0, 3)
        payload["status"] = "error"
        payload["metric"] = "ERROR"
        payload["error_message"] = str(e)
        payload["traceback_or_stderr_tail"] = traceback.format_exc(limit=5)
    finally:
        try:
            Path(temp_cfg).unlink(missing_ok=True)
        except Exception:
            pass

    if out_file.exists():
        try:
            raw = json.loads(out_file.read_text(encoding="utf-8"))
            scores = raw.get("task", {}).get("scores")
            enrich_from_task_scores(payload, scores)

            # optional pass-throughs from underlying runner
            payload["dataset_size"] = raw.get("dataset_size")
            if payload.get("train_wall_time_sec") is None:
                payload["train_wall_time_sec"] = raw.get("train_wall_time_sec")
            if raw.get("num_params") is not None:
                payload["num_params"] = raw.get("num_params")
        except Exception:
            pass

    if payload["status"] == "error":
        payload["metric_value"] = None

    finish_and_log(payload, date, task, model_config, note, out_file)


if __name__ == "__main__":
    main()
