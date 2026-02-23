#!/usr/bin/env python3
"""
True CHGNet structure-based finetuning runner for matbench_mp_e_form.
Optimized for "Quantum Leap" strategy:
1. Pretrained inference
2. Frozen backbone + AtomRef (Composition Correction) finetune
3. Full finetune
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Structure
from torch.utils.data import DataLoader, Dataset

# Imports from chgnet
try:
    from chgnet.model import CHGNet
except ImportError:
    CHGNet = None


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    freeze_backbone: bool = False
    mode: str = "finetune"  # finetune | pretrained


class MatbenchStructureDataset(Dataset):
    def __init__(self, structures: List[Structure], targets: List[float], model: CHGNet):
        self.structures = structures
        self.targets = targets
        self.converter = model.graph_converter
        # Cache graphs to speed up training
        print(f"Converting {len(structures)} structures to graphs...")
        self.graphs = [self.converter(s) for s in structures]

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor(self.targets[idx], dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_and_predict(
    train_structures: List[Structure],
    train_targets: List[float],
    test_structures: List[Structure],
    cfg: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    # Load model
    model = CHGNet.load()
    model.to(device)

    if cfg.mode == "pretrained" or cfg.epochs == 0:
        print("Running in pretrained inference mode...")
        model.eval()
    else:
        # Freeze backbone if requested
        if cfg.freeze_backbone:
            print("Freezing backbone (keeping only readout/composition layers trainable)...")
            # Heuristic: CHGNet has graph_comp blocks and final MLP. 
            # Composition model is also separate.
            for name, param in model.named_parameters():
                if any(x in name for x in ["readout", "composition", "mlp"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.HuberLoss() # Using Huber loss for robustness as per strategy

        # Data Loading
        try:
            from chgnet.data.loader import collate_graphs
        except ImportError:
            # Simple fallback collate if internal import fails
            def collate_graphs(batch):
                graphs, targets = zip(*batch)
                return list(graphs), torch.stack(targets)

        train_ds = MatbenchStructureDataset(train_structures, train_targets, model)
        
        def custom_collate(batch):
            gs, ts = zip(*batch)
            return collate_graphs(list(gs)), torch.stack(ts)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_collate)

        # Training loop
        model.train()
        print(f"Starting training for {cfg.epochs} epochs...")
        for epoch in range(cfg.epochs):
            t_start = time.perf_counter()
            losses = []
            for bg, targets in train_loader:
                bg = bg.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                # Forward: pass 'e' task
                preds = model(bg, task="e")
                # Preds is usually a dict if multiple tasks, or a tensor if single
                if isinstance(preds, dict):
                    pred_e = preds["e"]
                else:
                    pred_e = preds
                
                loss = criterion(pred_e.view(-1), targets.view(-1))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            t_end = time.perf_counter()
            print(f"Epoch {epoch+1}/{cfg.epochs} - Loss: {np.mean(losses):.6f} - Time: {t_end-t_start:.2f}s")

    # Prediction
    model.eval()
    print("Predicting on test structures...")
    all_preds = []
    with torch.no_grad():
        chunk_size = cfg.batch_size
        for i in range(0, len(test_structures), chunk_size):
            chunk = test_structures[i : i + chunk_size]
            # predict_structure is high-level API
            res = model.predict_structure(chunk, task="e", return_atom_refs=False)
            if isinstance(res, list):
                all_preds.extend([float(r["e"]) if isinstance(r, dict) else float(r) for r in res])
            else:
                val = float(res["e"]) if isinstance(res, dict) else float(res)
                all_preds.append(val)

    return np.array(all_preds)


def resolve_out_dir(cfg_raw: dict) -> Path:
    date_str = cfg_raw.get("date") or datetime.now().astimezone().strftime("%Y-%m-%d")
    run_name = cfg_raw.get("run_name", "chgnet_structure")
    out_root = Path(cfg_raw.get("output_root", "results/daily"))
    return out_root / date_str / run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    if CHGNet is None:
        raise RuntimeError(
            "CHGNet is not installed. Please install it first (e.g., `pip install chgnet`) "
            "and rerun this experiment."
        )

    t0 = time.perf_counter()
    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    
    seed = int(cfg_raw.get("seed", 42))
    set_seed(seed)

    task_name = cfg_raw["task"]["name"]
    train_dict = cfg_raw.get("training", {})
    extras = cfg_raw.get("training_extras", {})
    
    tcfg = TrainConfig(
        epochs=int(train_dict.get("epochs", 20)),
        batch_size=int(train_dict.get("batch_size", 64)),
        lr=float(train_dict.get("lr", 1e-4)),
        weight_decay=float(train_dict.get("weight_decay", 1e-5)),
        freeze_backbone=bool(extras.get("freeze_backbone", False)),
        mode=str(extras.get("mode", "finetune"))
    )

    device_cfg = cfg_raw.get("runtime", {}).get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and device_cfg == "auto" else "cpu")
    print(f"Using device: {device}")

    # Matbench Loading
    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    task = next(iter(mb.tasks))
    task.load()

    folds = cfg_raw.get("task", {}).get("folds", "all")
    folds_to_run = task.folds if folds == "all" else (folds if isinstance(folds, list) else [folds])

    fold_mae = {}
    
    for fold in folds_to_run:
        print(f"\n--- Starting Fold: {fold} ---")
        train_inputs, train_targets = task.get_train_and_val_data(fold)
        test_inputs = task.get_test_data(fold, include_target=False)
        
        preds = fit_and_predict(train_inputs, train_targets, test_inputs, tcfg, device)
        task.record(fold, preds)

        try:
            _, y_test = task.get_test_data(fold, include_target=True)
            mae = np.mean(np.abs(preds - y_test))
            fold_mae[fold] = float(mae)
            print(f"Fold {fold} MAE: {mae:.6f}")
        except Exception as e:
            print(f"Error calculating fold MAE: {e}")

    # Save results
    out_dir = resolve_out_dir(cfg_raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_mae = np.mean(list(fold_mae.values())) if fold_mae else None
    
    result = {
        "task": {
            "name": task_name,
            "scores": task.scores,
        },
        "metrics": {
            "metric_name": "MAE",
            "metric_value": mean_mae,
            "fold_scores": fold_mae,
            "mean": mean_mae,
            "std": np.std(list(fold_mae.values())) if len(fold_mae) > 1 else 0.0,
        },
        "model": {
            "name": "CHGNet-Structure",
            "mode": tcfg.mode,
            "freeze_backbone": tcfg.freeze_backbone,
        },
        "hparams": train_dict,
        "runtime": {"device": str(device), "total_time_sec": time.perf_counter() - t0},
        "seed": seed
    }
    
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nFinal mean MAE: {mean_mae}")
    print(f"Wrote results to {out_file}")
    
    if mean_mae is not None:
        print(f"FINAL_METRIC_MAE={mean_mae:.6f}")
    else:
        print("FINAL_METRIC_MAE=NA")

if __name__ == "__main__":
    main()
