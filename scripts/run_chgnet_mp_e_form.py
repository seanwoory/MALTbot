#!/usr/bin/env python3
"""Route-B baseline runner for matbench_mp_e_form.

- Uses matbench record() API for official scoring.
- Computes explicit per-fold/aggregate MAE for stable downstream logging.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Element, Structure
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    dropout: float


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def structure_to_vec(structure: Structure, n_elements: int = 118) -> np.ndarray:
    comp = structure.composition.fractional_composition
    vec = np.zeros(n_elements, dtype=np.float32)
    for el, frac in comp.get_el_amt_dict().items():
        z = Element(el).Z
        if 1 <= z <= n_elements:
            vec[z - 1] = float(frac)
    return vec


def structures_to_matrix(structures: Iterable[Structure], n_elements: int = 118) -> np.ndarray:
    return np.stack([structure_to_vec(s, n_elements=n_elements) for s in structures], axis=0)


def fit_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    model = MLPRegressor(in_dim=x_train.shape[1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(x_test).to(device)).cpu().numpy().astype(float)
    return preds, int(num_params)


def resolve_out_dir(cfg_raw: dict) -> Path:
    # Canonical preferred: results/daily/<DATE>/<BATCH>/<RUN_NAME>
    date_str = cfg_raw.get("date") or datetime.now().astimezone().strftime("%Y-%m-%d")
    run_name = cfg_raw.get("run_name", "chgnet_mp_e_form_route_b")
    out_root = Path(cfg_raw.get("output_root", "results/daily"))

    # If output_root already includes date/batch, avoid re-appending date accidentally.
    # We standardize by always creating out_root/date/run_name.
    return out_root / date_str / run_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chgnet_mp_e_form.yaml")
    args = ap.parse_args()

    t0 = time.perf_counter()

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg_raw["seed"])
    set_seed(seed)

    task_name = cfg_raw["task"]["name"]
    n_elements = int(cfg_raw["features"].get("n_elements", 118))
    tcfg = TrainConfig(**cfg_raw["training"])

    device_cfg = cfg_raw["runtime"].get("device", "auto")
    device = torch.device("cuda" if (device_cfg == "auto" and torch.cuda.is_available()) else device_cfg if device_cfg != "auto" else "cpu")

    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    task = next(iter(mb.tasks))
    task.load()

    folds = cfg_raw["task"].get("folds", "all")
    folds_to_run: List[str] = task.folds if folds == "all" else list(folds)

    dataset_size: dict[str, dict[str, int]] = {}
    fold_mae: dict[str, float] = {}
    model_num_params: int | None = None

    for fold in folds_to_run:
        train_inputs, train_targets = task.get_train_and_val_data(fold)
        test_inputs = task.get_test_data(fold, include_target=False)

        x_train = structures_to_matrix(train_inputs, n_elements=n_elements)
        y_train = np.asarray(train_targets, dtype=np.float32)
        x_test = structures_to_matrix(test_inputs, n_elements=n_elements)

        dataset_size[fold] = {"train_val": int(len(train_inputs)), "test": int(len(test_inputs))}

        preds, n_params = fit_and_predict(x_train, y_train, x_test, tcfg, device)
        model_num_params = n_params if model_num_params is None else model_num_params
        task.record(fold, preds.tolist())

        # Best-effort fold MAE using available labels (if provided by local matbench data).
        try:
            _, y_test = task.get_test_data(fold, include_target=True)
            y_true = np.asarray(y_test, dtype=np.float32)
            if len(y_true) == len(preds):
                fold_mae[fold] = float(np.mean(np.abs(preds - y_true)))
        except Exception:
            pass

    out_dir = resolve_out_dir(cfg_raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fold_mae:
        mean_mae = float(np.mean(list(fold_mae.values())))
        std_mae = float(np.std(list(fold_mae.values()))) if len(fold_mae) > 1 else 0.0
    else:
        mean_mae = None
        std_mae = None

    # Keep raw task.scores for compatibility, but provide explicit MAE structure.
    task_scores_raw = None
    try:
        task_scores_raw = task.scores
    except Exception:
        pass

    result = {
        "task": {
            "name": task_name,
            "scores": {
                "mae_mean": mean_mae,
                "mae_std": std_mae,
                "fold_mae": fold_mae or None,
                "raw": task_scores_raw,
            },
        },
        "metrics": {
            "metric_name": "MAE",
            "metric_value": mean_mae,
            "metric_unit": "eV/atom" if task_name == "matbench_mp_e_form" else "eV",
            "fold_scores": fold_mae or None,
            "mean": mean_mae,
            "std": std_mae,
        },
        "model": {
            "name": "Route-B CHGNet-lite (composition MLP baseline)",
            "note": "Minimal reproducible baseline. Replace with full CHGNet finetune for final Route-B.",
        },
        "hparams": cfg_raw["training"],
        "seed": seed,
        "num_params": model_num_params,
        "dataset_size": dataset_size,
        "train_wall_time_sec": round(time.perf_counter() - t0, 3),
        "runtime": {
            "device": str(device),
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }

    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote: {out_file}")
    if mean_mae is not None:
        print(f"FINAL_METRIC_MAE={mean_mae:.6f}")
    else:
        print("FINAL_METRIC_MAE=NA")


if __name__ == "__main__":
    main()
