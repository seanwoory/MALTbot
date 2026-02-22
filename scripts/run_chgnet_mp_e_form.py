#!/usr/bin/env python3
"""Route-B baseline runner for matbench_mp_e_form.

- Uses matbench record() API for official scoring.
- Applies configurable training levers (target transform / LR scheduler / EMA).
- Emits stable metrics and FINAL_METRIC_MAE marker for orchestration.
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
    target_transform: str = "none"  # none|standardize
    scheduler: str = "none"  # none|cosine|step
    step_size: int = 20
    gamma: float = 0.5
    ema: bool = False
    ema_decay: float = 0.995
    freeze_backbone: bool = False
    tta_samples: int = 1
    tta_noise_std: float = 0.0
    ensemble_seeds: List[int] | None = None


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


def parse_mae_from_task_scores(raw_scores):
    """Best-effort parser for matbench task.scores structures."""
    fold_mae = {}
    mean_mae = None
    std_mae = None

    if isinstance(raw_scores, dict):
        # Common shape: {"mae": {"mean": ..., "std": ..., "fold_0": ...}}
        mae_obj = raw_scores.get("mae")
        if isinstance(mae_obj, dict):
            for k, v in mae_obj.items():
                if isinstance(v, (int, float)):
                    if k.startswith("fold"):
                        fold_mae[k] = float(v)
                    elif k == "mean":
                        mean_mae = float(v)
                    elif k == "std":
                        std_mae = float(v)
        elif isinstance(mae_obj, (int, float)):
            mean_mae = float(mae_obj)

        # Alternative shape: top-level fold keys
        if not fold_mae:
            for k, v in raw_scores.items():
                if isinstance(v, (int, float)) and str(k).startswith("fold"):
                    fold_mae[str(k)] = float(v)

    if fold_mae and mean_mae is None:
        vals = list(fold_mae.values())
        mean_mae = float(np.mean(vals))
        std_mae = float(np.std(vals)) if len(vals) > 1 else 0.0

    # Last fallback: first numeric anywhere
    if mean_mae is None:
        stack = [raw_scores]
        while stack:
            cur = stack.pop()
            if isinstance(cur, (int, float)):
                mean_mae = float(cur)
                break
            if isinstance(cur, dict):
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)

    return fold_mae or None, mean_mae, std_mae


def fit_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    model = MLPRegressor(in_dim=x_train.shape[1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    # target transform
    y_mu, y_sigma = 0.0, 1.0
    y_train_work = y_train.astype(np.float32)
    if cfg.target_transform == "standardize":
        y_mu = float(np.mean(y_train_work))
        y_sigma = float(np.std(y_train_work))
        if y_sigma < 1e-8:
            y_sigma = 1.0
        y_train_work = (y_train_work - y_mu) / y_sigma

    criterion = nn.L1Loss()

    if cfg.freeze_backbone:
        # Backbone: first two Linear blocks, Head: final Linear layer.
        for i, m in enumerate(model.net):
            if isinstance(m, nn.Linear) and i < 6:
                for p in m.parameters():
                    p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    elif cfg.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.step_size), gamma=cfg.gamma)

    ema_state = None
    if cfg.ema:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_work))
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

            if ema_state is not None:
                with torch.no_grad():
                    msd = model.state_dict()
                    for k in ema_state:
                        ema_state[k].mul_(cfg.ema_decay).add_(msd[k].detach(), alpha=1.0 - cfg.ema_decay)

        if scheduler is not None:
            scheduler.step()

    if ema_state is not None:
        model.load_state_dict(ema_state)

    model.eval()
    with torch.no_grad():
        x_test_t = torch.from_numpy(x_test).to(device)
        if cfg.tta_samples > 1 and cfg.tta_noise_std > 0:
            pred_acc = torch.zeros(x_test_t.shape[0], device=device)
            for _ in range(cfg.tta_samples):
                noise = torch.randn_like(x_test_t) * float(cfg.tta_noise_std)
                pred_acc += model(x_test_t + noise)
            preds = (pred_acc / float(cfg.tta_samples)).cpu().numpy().astype(float)
        else:
            preds = model(x_test_t).cpu().numpy().astype(float)

    if cfg.target_transform == "standardize":
        preds = preds * y_sigma + y_mu

    return preds, int(num_params)


def resolve_out_dir(cfg_raw: dict) -> Path:
    date_str = cfg_raw.get("date") or datetime.now().astimezone().strftime("%Y-%m-%d")
    run_name = cfg_raw.get("run_name", "chgnet_mp_e_form_route_b")
    out_root = Path(cfg_raw.get("output_root", "results/daily"))
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

    extras = cfg_raw.get("training_extras", {}) or {}
    train_dict = dict(cfg_raw["training"])
    tcfg = TrainConfig(
        epochs=int(train_dict["epochs"]),
        batch_size=int(train_dict["batch_size"]),
        lr=float(train_dict["lr"]),
        weight_decay=float(train_dict["weight_decay"]),
        hidden_dim=int(train_dict["hidden_dim"]),
        dropout=float(train_dict["dropout"]),
        target_transform=str(extras.get("target_transform", "none")),
        scheduler=str(extras.get("scheduler", "none")),
        step_size=int(extras.get("step_size", 20)),
        gamma=float(extras.get("gamma", 0.5)),
        ema=bool(extras.get("ema", False)),
        ema_decay=float(extras.get("ema_decay", 0.995)),
        tta_samples=max(1, int(extras.get("tta_samples", 1))),
        tta_noise_std=max(0.0, float(extras.get("tta_noise_std", 0.0))),
        ensemble_seeds=[int(s) for s in extras.get("ensemble_seeds", [])] or None,
    )

    device_cfg = cfg_raw["runtime"].get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    task = next(iter(mb.tasks))
    task.load()

    folds = cfg_raw["task"].get("folds", "all")
    folds_to_run: List[str] = task.folds if folds == "all" else list(folds)

    dataset_size: dict[str, dict[str, int]] = {}
    model_num_params: int | None = None

    for fold in folds_to_run:
        train_inputs, train_targets = task.get_train_and_val_data(fold)
        test_inputs = task.get_test_data(fold, include_target=False)

        x_train = structures_to_matrix(train_inputs, n_elements=n_elements)
        y_train = np.asarray(train_targets, dtype=np.float32)
        x_test = structures_to_matrix(test_inputs, n_elements=n_elements)

        dataset_size[fold] = {
            "train_val": int(len(train_inputs)),
            "test": int(len(test_inputs)),
        }

        if tcfg.ensemble_seeds and len(tcfg.ensemble_seeds) > 1:
            pred_members: list[np.ndarray] = []
            n_params = None
            for member_seed in tcfg.ensemble_seeds:
                set_seed(int(member_seed))
                preds_i, n_params_i = fit_and_predict(x_train, y_train, x_test, tcfg, device)
                pred_members.append(preds_i)
                if n_params is None:
                    n_params = n_params_i
            preds = np.mean(np.stack(pred_members, axis=0), axis=0)
            model_num_params = n_params if model_num_params is None else model_num_params
        else:
            preds, n_params = fit_and_predict(x_train, y_train, x_test, tcfg, device)
            model_num_params = n_params if model_num_params is None else model_num_params

        task.record(fold, preds.tolist())

    out_dir = resolve_out_dir(cfg_raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_scores = None
    fold_mae, mean_mae, std_mae = None, None, None
    try:
        raw_scores = task.scores
        fold_mae, mean_mae, std_mae = parse_mae_from_task_scores(raw_scores)
    except Exception:
        pass

    result = {
        "task": {
            "name": task_name,
            "scores": {
                "mae_mean": mean_mae,
                "mae_std": std_mae,
                "fold_mae": fold_mae,
                "raw": raw_scores,
            },
        },
        "metrics": {
            "metric_name": "MAE",
            "metric_value": mean_mae,
            "metric_unit": "eV/atom" if task_name == "matbench_mp_e_form" else "eV",
            "fold_scores": fold_mae,
            "mean": mean_mae,
            "std": std_mae,
        },
        "model": {
            "name": "Route-B CHGNet-lite (composition MLP baseline)",
            "note": "Minimal reproducible baseline. Replace with full CHGNet finetune for final Route-B.",
        },
        "hparams": cfg_raw["training"],
        "training_extras": extras,
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
    if isinstance(mean_mae, (int, float)):
        print(f"FINAL_METRIC_MAE={float(mean_mae):.6f}")
    else:
        print("FINAL_METRIC_MAE=NA")


if __name__ == "__main__":
    main()
