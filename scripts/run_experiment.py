#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
MARKER_RE = re.compile(r"FINAL_METRIC_MAE=([0-9]+(?:\.[0-9]+)?)")


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
    p = Path(args.exp) if args.exp else REPO / "configs" / "experiments" / f"{args.exp_name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Experiment config not found: {p}")
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}), p


def append_results_md(line: str) -> None:
    with (REPO / "RESULTS.md").open("a", encoding="utf-8") as fp:
        fp.write("\n" + line)


def base_payload(exp_name: str, task: str, seed: int, model_config: str, note: str, exp_path: Path, out_file: Path) -> dict[str, Any]:
    cuda_available, gpu_type, torch_ver = gpu_info()
    return {
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
        "exp_params": {},
        "output_path": str(out_file.relative_to(REPO)),
        "env": {
            "python": sys.version,
            "git_commit": git_commit(),
            "torch": torch_ver,
            "cuda_available": cuda_available,
            "gpu_type": gpu_type,
        },
    }


def enrich_metric(payload: dict[str, Any], raw: dict[str, Any], stdout: str) -> None:
    # 1) direct normalized metrics from child output if present
    metrics = raw.get("metrics") if isinstance(raw, dict) else None
    if isinstance(metrics, dict):
        payload["metric_name"] = metrics.get("metric_name", payload["metric_name"])
        payload["metric_unit"] = metrics.get("metric_unit", payload["metric_unit"])
        payload["metric_value"] = metrics.get("metric_value")
        payload["fold_scores"] = metrics.get("fold_scores")
        payload["mean"] = metrics.get("mean")
        payload["std"] = metrics.get("std")

    # 2) raw task.scores fallback
    scores = (raw.get("task") or {}).get("scores") if isinstance(raw, dict) else None
    if scores is not None:
        payload.setdefault("task", {})["scores"] = scores
    if payload.get("fold_scores") is None and isinstance(scores, dict):
        payload["fold_scores"] = scores.get("fold_mae") or scores.get("fold_scores")

    if payload.get("metric_value") is None:
        payload["metric_value"] = first_numeric(scores)

    # 3) stdout marker fallback
    if payload.get("metric_value") is None and stdout:
        m = MARKER_RE.search(stdout)
        if m:
            payload["metric_value"] = float(m.group(1))

    # derive mean/std if fold scores available but mean/std missing
    vals: list[float] = []
    if payload.get("fold_scores") is not None:
        collect_numerics(payload["fold_scores"], vals)
    if vals:
        if payload.get("mean") is None:
            payload["mean"] = statistics.fmean(vals)
        if payload.get("std") is None:
            payload["std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    mv = payload.get("metric_value")
    if isinstance(mv, (int, float)):
        payload["metric"] = float(mv)
    elif payload.get("status") == "success":
        payload["metric"] = "RECORDED"


def locate_child_output(canonical_out: Path, date: str, batch: str, exp_name: str, stdout: str) -> Path | None:
    if canonical_out.exists():
        return canonical_out

    # Backward-compatible old nested layout.
    nested = REPO / "results" / "daily" / date / batch / date / exp_name / "results.json"
    if nested.exists():
        return nested

    # Try to parse `Wrote: <path>` line.
    for line in (stdout or "").splitlines()[::-1]:
        if line.startswith("Wrote: "):
            p = Path(line.replace("Wrote: ", "").strip())
            p = p if p.is_absolute() else (REPO / p)
            if p.exists():
                return p

    return None


def finish_and_log(payload: dict[str, Any], date: str, task: str, model_config: str, note: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metric_for_line = payload.get("metric")
    if isinstance(metric_for_line, (int, float)):
        metric_for_line = f"{float(metric_for_line):.6f}"
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
    task_obj = exp.get("task", "matbench_mp_e_form")
    task = task_obj.get("name", "matbench_mp_e_form") if isinstance(task_obj, dict) else task_obj
    seed = int(exp.get("seed", 42))
    model_config = exp.get("model_config", exp_name)
    note = exp.get("note", "")
    enabled = bool(exp.get("enabled", True))
    runner = exp.get("runner", "chgnet")

    date = os.getenv("MALTBOT_DATE", kst_today())
    batch = os.getenv("MALTBOT_BATCH_RUN_NAME", "batch")

    out_file = REPO / "results" / "daily" / date / batch / exp_name / "results.json"
    payload = base_payload(exp_name, task, seed, model_config, note, exp_path, out_file)
    payload["exp_params"] = exp.get("params") or {}

    if not enabled:
        payload["status"] = "skipped"
        payload["metric"] = "SKIPPED"
        payload["error_message"] = "Experiment disabled in YAML (enabled: false)."
        finish_and_log(payload, date, task, model_config, note, out_file)
        return

    supported_runners = {
        "chgnet",
        "chgnet_tta4",
        "ensemble_3seed",
        "chgnet_ensemble3",
        "mlp_pretrained_infer_fallback",
        "mlp_head_finetune_freeze",
        "chgnet_structure",
        # Reserved names for future true CHGNet implementations:
        "chgnet_pretrained_infer",
        "chgnet_head_finetune_freeze",
    }
    if runner not in supported_runners:
        payload["status"] = "skipped"
        payload["metric"] = "SKIPPED"
        payload["error_message"] = f"Runner '{runner}' not implemented yet."
        finish_and_log(payload, date, task, model_config, note, out_file)
        return

    base_cfg = yaml.safe_load((REPO / "configs" / "chgnet_mp_e_form.yaml").read_text(encoding="utf-8"))
    base_cfg["seed"] = seed
    base_cfg["run_name"] = f"{batch}/{exp_name}"  # ensure canonical non-nested path
    base_cfg["date"] = date
    base_cfg["output_root"] = "results/daily"
    base_cfg.setdefault("task", {})["name"] = task
    if isinstance(task_obj, dict) and "folds" in task_obj:
        base_cfg.setdefault("task", {})["folds"] = task_obj["folds"]

    # Apply experiment-specific levers from YAML params.
    params = exp.get("params") or {}
    tr = base_cfg.setdefault("training", {})
    extras = base_cfg.setdefault("training_extras", {})
    if "lr" in params:
        tr["lr"] = float(params["lr"])
    if "epochs" in params:
        tr["epochs"] = int(params["epochs"])
    if "weight_decay" in params:
        tr["weight_decay"] = float(params["weight_decay"])
    if "target_transform" in params:
        extras["target_transform"] = str(params["target_transform"])
    if "scheduler" in params:
        extras["scheduler"] = str(params["scheduler"])
    if "step_size" in params:
        extras["step_size"] = int(params["step_size"])
    if "gamma" in params:
        extras["gamma"] = float(params["gamma"])
    if "ema" in params:
        extras["ema"] = bool(params["ema"])
    if "ema_decay" in params:
        extras["ema_decay"] = float(params["ema_decay"])
    if "tta_samples" in params:
        extras["tta_samples"] = int(params["tta_samples"])
    if "tta_noise_std" in params:
        extras["tta_noise_std"] = float(params["tta_noise_std"])
    if "data_fraction" in params:
        extras["data_fraction"] = float(params["data_fraction"])
    if "cache_fraction" in params:
        extras["cache_fraction"] = float(params["cache_fraction"])
    if "train_fraction" in params:
        extras["train_fraction"] = float(params["train_fraction"])
    if "val_fraction" in params:
        extras["val_fraction"] = float(params["val_fraction"])
    if "early_stopping_patience" in params:
        extras["early_stopping_patience"] = int(params["early_stopping_patience"])
    if "folds" in params:
        f = params["folds"]
        if isinstance(f, str):
            base_cfg.setdefault("task", {})["folds"] = [f]
        elif isinstance(f, list):
            base_cfg.setdefault("task", {})["folds"] = f

    if "seeds" in params and isinstance(params["seeds"], list):
        extras["ensemble_seeds"] = [int(s) for s in params["seeds"]]

    if runner == "chgnet_tta4":
        extras.setdefault("tta_samples", 4)
        extras.setdefault("tta_noise_std", 0.01)

    if runner in {"ensemble_3seed", "chgnet_ensemble3"}:
        extras.setdefault("ensemble_seeds", [41, 42, 43])

    if runner == "mlp_pretrained_infer_fallback":
        # Explicitly MLP fallback (not true CHGNet pretrained inference).
        tr["epochs"] = int(params.get("fallback_epochs", 20))
        extras.setdefault("mode", "mlp_pretrained_infer_fallback")

    if runner == "mlp_head_finetune_freeze":
        extras["freeze_backbone"] = True
        tr["epochs"] = int(params.get("epochs", tr.get("epochs", 30)))
        if bool(params.get("partial_fold0_only", True)):
            base_cfg.setdefault("task", {})["folds"] = ["fold_0"]
            payload["status"] = "partial"

    target_script = "scripts/run_chgnet_mp_e_form.py"
    if runner in {"chgnet_structure", "chgnet_pretrained_infer", "chgnet_head_finetune_freeze", "chgnet_ensemble3"}:
        target_script = "scripts/run_chgnet_structure.py"

    if runner == "chgnet_pretrained_infer":
        tr["epochs"] = 0
        extras["mode"] = "pretrained"

    if runner == "chgnet_head_finetune_freeze":
        extras["freeze_backbone"] = True
        tr["epochs"] = int(params.get("epochs", 10))
        tr["lr"] = float(params.get("lr", 1e-3))

    if runner == "chgnet_ensemble3":
        extras["ensemble_seeds"] = params.get("seeds", [42, 43, 44])
        tr["epochs"] = int(params.get("epochs", 20))
        # Map specific params for structure runner
        if "freeze_backbone" in params:
            extras["freeze_backbone"] = bool(params["freeze_backbone"])

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(base_cfg, tf, sort_keys=False)
        temp_cfg = tf.name

    t0 = time.perf_counter()
    stdout = ""
    stderr_tail = ""
    try:
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-u", target_script, "--config", temp_cfg],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )

        out_lines: list[str] = []
        if proc.stdout is not None:
            for line in proc.stdout:
                print(line, end="")  # real-time streaming to caller (e.g., Colab)
                out_lines.append(line)

        returncode = proc.wait()
        stdout = "".join(out_lines)
        stderr_tail = "\n".join(stdout.splitlines()[-40:])

        payload["train_wall_time_sec"] = round(time.perf_counter() - t0, 3)
        payload["run"] = {
            "returncode": returncode,
            "stdout_tail": "\n".join(stdout.splitlines()[-40:]),
            "stderr_tail": stderr_tail,
        }
        if returncode != 0:
            payload["status"] = "error"
            payload["metric"] = "ERROR"
            payload["error_message"] = f"runner exited with code {returncode}"
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

    child_path = locate_child_output(out_file, date, batch, exp_name, stdout)
    if child_path is not None:
        try:
            raw = json.loads(child_path.read_text(encoding="utf-8"))
            enrich_metric(payload, raw, stdout)
            payload["dataset_size"] = raw.get("dataset_size")
            payload["num_params"] = raw.get("num_params")
            if payload.get("train_wall_time_sec") is None:
                payload["train_wall_time_sec"] = raw.get("train_wall_time_sec")
            # keep canonical output path even if child was nested old path
            payload["source_output_path"] = str(child_path.relative_to(REPO))
        except Exception:
            pass

    if payload["status"] == "error":
        payload["metric_value"] = None

    # ensure success lines are numeric whenever we extracted one
    if payload["status"] == "success" and isinstance(payload.get("metric_value"), (int, float)):
        payload["metric"] = float(payload["metric_value"])

    finish_and_log(payload, date, task, model_config, note, out_file)


if __name__ == "__main__":
    main()
