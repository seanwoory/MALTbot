#!/usr/bin/env python3
"""Route-B baseline runner for matbench_mp_e_form.

- Uses matbench record() API for official scoring.
- Provides a minimal PyTorch regression pipeline on composition features.
- Script name keeps Route-B intent for CHGNet finetuning, while remaining
  lightweight/reproducible in Colab.
"""

from __future__ import annotations

import argparse
import json
import random
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


def _ensure_series(obj):
    try:
        import pandas as pd  # noqa: F401

        return obj
    except Exception:
        return obj


def fit_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    model = MLPRegressor(in_dim=x_train.shape[1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
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
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chgnet_mp_e_form.yaml")
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg_raw["seed"])
    set_seed(seed)

    task_name = cfg_raw["task"]["name"]
    n_elements = int(cfg_raw["features"].get("n_elements", 118))

    tcfg = TrainConfig(**cfg_raw["training"])

    device_cfg = cfg_raw["runtime"].get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    task = next(iter(mb.tasks))
    
    folds = cfg_raw["task"].get("folds", "all")
    folds_to_run: List[str] = task.folds if folds == "all" else list(folds)

    for fold in folds_to_run:
        train_inputs, train_targets = task.get_train_and_val_data(fold)
        test_inputs = task.get_test_data(fold, include_target=False)

        x_train = structures_to_matrix(train_inputs, n_elements=n_elements)
        y_train = np.asarray(train_targets, dtype=np.float32)
        x_test = structures_to_matrix(test_inputs, n_elements=n_elements)

        preds = fit_and_predict(x_train, y_train, x_test, tcfg, device)
        task.record(fold, preds.tolist())

    date_str = datetime.now().astimezone().strftime("%Y-%m-%d")
    run_name = cfg_raw.get("run_name", "chgnet_mp_e_form_route_b")
    out_root = Path(cfg_raw.get("output_root", "results/daily"))
    out_dir = out_root / date_str / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "task": {
            "name": task_name,
            "scores": task.scores,
        },
        "model": {
            "name": "Route-B CHGNet-lite (composition MLP baseline)",
            "note": "Minimal reproducible baseline. Replace with full CHGNet finetune for final Route-B.",
        },
        "hparams": cfg_raw["training"],
        "seed": seed,
        "runtime": {
            "device": str(device),
            "python": __import__("sys").version,
            "torch": torch.__version__,
        },
    }

    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote: {out_file}")


if __name__ == "__main__":
    main()
