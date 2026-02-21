#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_MD = REPO_ROOT / "RESULTS.md"
ASSET_DIR = REPO_ROOT / "docs" / "assets"

TOP5 = {
    "matbench_mp_e_form": [
        ("Top-1 coGN", 0.0170),
        ("Top-2 coNGN", 0.0178),
        ("Top-3 ALIGNN", 0.0215),
        ("Top-4 SchNet", 0.0218),
        ("Top-5 DimeNet++", 0.0235),
    ],
    "matbench_mp_gap": [
        ("Top-1 coGN", 0.1559),
        ("Top-2 DeeperGATGNN", 0.1694),
        ("Top-3 coNGN", 0.1697),
        ("Top-4 ALIGNN", 0.1861),
        ("Top-5 MegNet", 0.1934),
    ],
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Row:
    date: str
    task: str
    model: str
    metric: float | None
    metric_raw: str
    note: str
    path: str


def first_numeric(obj: Any) -> float | None:
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            n = first_numeric(v)
            if n is not None:
                return n
    if isinstance(obj, list):
        for v in obj:
            n = first_numeric(v)
            if n is not None:
                return n
    return None


def parse_metric(metric_field: str) -> float | None:
    m = NUM_RE.search(metric_field)
    return float(m.group(0)) if m else None


def maybe_metric_from_json(path_str: str) -> float | None:
    p = REPO_ROOT / path_str
    if not p.exists() or p.suffix.lower() != ".json":
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    scores = data.get("task", {}).get("scores")
    return first_numeric(scores)


def parse_results_md() -> list[Row]:
    rows: list[Row] = []
    if not RESULTS_MD.exists():
        return rows

    for raw in RESULTS_MD.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        date, task, model, metric_field, note, path = parts[:6]
        if not DATE_RE.match(date):
            continue

        metric = parse_metric(metric_field)
        if metric is None and any(x in metric_field.upper() for x in ["TBD", "ERROR"]):
            metric = maybe_metric_from_json(path)

        rows.append(Row(date, task, model, metric, metric_field, note, path))

    return rows


def render_task_svg(task: str, rows: list[Row]) -> Path:
    task_rows = [r for r in rows if r.task == task]
    task_rows.sort(key=lambda r: (r.date, r.model, r.path))

    out = ASSET_DIR / f"progress_{task}.svg"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    x_labels = [r.date for r in task_rows]
    y_vals = [r.metric for r in task_rows]

    valid_idx = [i for i, y in enumerate(y_vals) if y is not None]
    if valid_idx:
        xs = [i for i in valid_idx]
        ys = [y_vals[i] for i in valid_idx]
        ax.plot(xs, ys, marker="o", linewidth=2.2, color="#1f77b4", label="Our daily run")
        for i in xs:
            ax.annotate(task_rows[i].model, (i, y_vals[i]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No numeric metric found yet", transform=ax.transAxes, ha="center", va="center")

    for name, v in TOP5.get(task, []):
        ax.axhline(v, linestyle="--", linewidth=1.2, alpha=0.85, label=f"{name}: {v:.4f}")

    ax.set_title(f"Progress — {task} (MAE lower is better)", fontsize=13)
    ax.set_xlabel("Run date")
    ax.set_ylabel("MAE")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()

    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rows = parse_results_md()

    targets = ["matbench_mp_e_form", "matbench_mp_gap"]
    generated: list[str] = []
    for task in targets:
        p = render_task_svg(task, rows)
        generated.append(str(p.relative_to(REPO_ROOT)))

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": str(RESULTS_MD.relative_to(REPO_ROOT)),
        "targets": targets,
        "num_rows": len(rows),
        "generated_files": generated,
    }
    (ASSET_DIR / "progress_data.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Generated:", generated)


if __name__ == "__main__":
    main()
