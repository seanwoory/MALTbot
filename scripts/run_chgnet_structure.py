#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Structure
from torch.utils.data import DataLoader, Dataset

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def structure_key(s: Structure) -> str:
    # Deterministic key for cross-fold graph cache.
    return json.dumps(s.as_dict(), sort_keys=True, ensure_ascii=False)


def _build_converter_with_cutoff(base_converter, cutoff: float):
    conv_cls = type(base_converter)
    try:
        sig = inspect.signature(conv_cls.__init__)
        kwargs = {}
        for name in sig.parameters:
            if name == "self":
                continue
            if hasattr(base_converter, name):
                kwargs[name] = getattr(base_converter, name)
        for key in ("atom_graph_cutoff", "cutoff", "radius"):
            if key in sig.parameters:
                kwargs[key] = cutoff
        return conv_cls(**kwargs)
    except Exception:
        return conv_cls(atom_graph_cutoff=cutoff)


def convert_structure_with_adaptive_cutoff(structure: Structure, converter, idx: int = -1):
    try:
        return converter(structure), None
    except ValueError as e:
        if "isolated atom" not in str(e).lower():
            raise

    for cutoff in (10.0, 20.0):
        try:
            retry_converter = _build_converter_with_cutoff(converter, cutoff)
            g = retry_converter(structure)
            print(f"[WARN] Adaptive cutoff used for structure idx={idx}: cutoff={cutoff}")
            return g, cutoff
        except ValueError as e:
            if "isolated atom" not in str(e).lower():
                raise
            continue

    raise RuntimeError(
        f"Graph conversion failed for structure idx={idx} after adaptive cutoffs (10.0, 20.0)."
    )


class GraphDataset(Dataset):
    def __init__(self, graphs: list, targets: list[float]):
        self.graphs = graphs
        self.targets = targets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor(self.targets[idx], dtype=torch.float32)


def collate_or_fallback(batch):
    graphs, targets = zip(*batch)
    try:
        from chgnet.data.loader import collate_graphs

        return collate_graphs(list(graphs)), torch.stack(targets)
    except Exception:
        return list(graphs), torch.stack(targets)


def run_predict_on_graphs(model, graphs: list, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds: list[float] = []
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            chunk_graphs = graphs[i : i + batch_size]
            if hasattr(model, "predict_graph"):
                out = model.predict_graph(chunk_graphs, task="e")
                if isinstance(out, list):
                    for r in out:
                        if isinstance(r, dict):
                            v = r.get("e", r.get("energy"))
                            if v is None:
                                raise KeyError("predict_graph dict response missing 'e'/'energy'")
                            preds.append(float(v))
                        else:
                            preds.append(float(r))
                elif isinstance(out, dict):
                    v = out.get("e", out.get("energy"))
                    if v is None:
                        raise KeyError("predict_graph dict response missing 'e'/'energy'")
                    preds.append(float(v))
                else:
                    preds.extend([float(x) for x in np.array(out).reshape(-1)])
            else:
                bg, _ = collate_or_fallback([(g, 0.0) for g in chunk_graphs])
                if hasattr(bg, "to"):
                    bg = bg.to(device)
                out = model(bg, task="e")
                pe = out["e"] if isinstance(out, dict) else out
                preds.extend([float(x) for x in np.array(pe.detach().cpu()).reshape(-1)])
    return np.array(preds)


def train_one_fold(
    model,
    train_graphs: list,
    train_targets: list[float],
    cfg: TrainConfig,
    device: torch.device,
    history_csv: Path,
    ckpt_path: Path,
) -> None:
    if cfg.mode == "pretrained" or cfg.epochs == 0:
        return

    if cfg.freeze_backbone:
        print("Freezing backbone (keeping readout/composition/mlp trainable)...")
        for name, param in model.named_parameters():
            if any(x in name for x in ["readout", "composition", "mlp"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    n = len(train_graphs)
    idx = np.random.permutation(n)
    val_n = max(1, int(0.1 * n))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]
    if len(tr_idx) == 0:
        tr_idx = val_idx

    tr_graphs = [train_graphs[i] for i in tr_idx]
    tr_targets = [train_targets[i] for i in tr_idx]
    va_graphs = [train_graphs[i] for i in val_idx]
    va_targets = [train_targets[i] for i in val_idx]

    train_loader = DataLoader(
        GraphDataset(tr_graphs, tr_targets),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_or_fallback,
    )

    criterion = nn.HuberLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_mae = float("inf")
    history_rows = []
    history_csv.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for bg, y in train_loader:
            if hasattr(bg, "to"):
                bg = bg.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(bg, task="e")
            pe = out["e"] if isinstance(out, dict) else out
            loss = criterion(pe.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else None
        val_pred = run_predict_on_graphs(model, va_graphs, device, cfg.batch_size)
        val_true = np.array(va_targets, dtype=float)
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
        val_loss = float(np.mean(np.abs(val_pred - val_true)))

        history_rows.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae})
        print(f"Epoch {ep}/{cfg.epochs} train_loss={train_loss:.6f} val_mae={val_mae:.6f}")

        if val_mae < best_mae:
            best_mae = val_mae
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    with history_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae"])
        w.writeheader()
        w.writerows(history_rows)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))


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
        raise RuntimeError("CHGNet is not installed. Please install it first (`pip install chgnet`).")

    t0 = time.perf_counter()
    cfg_raw = yaml.safe_load(Path(args.config).read_text())

    seed = int(cfg_raw.get("seed", 42))
    set_seed(seed)

    task_name = cfg_raw["task"]["name"]
    train_dict = cfg_raw.get("training", {})
    extras = cfg_raw.get("training_extras", {})

    cfg = TrainConfig(
        epochs=int(train_dict.get("epochs", 20)),
        batch_size=int(train_dict.get("batch_size", 64)),
        lr=float(train_dict.get("lr", 1e-4)),
        weight_decay=float(train_dict.get("weight_decay", 1e-5)),
        freeze_backbone=bool(extras.get("freeze_backbone", False)),
        mode=str(extras.get("mode", "finetune")),
    )

    device_cfg = cfg_raw.get("runtime", {}).get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and device_cfg == "auto" else "cpu")
    print(f"Using device: {device}")

    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    task = next(iter(mb.tasks))
    task.load()

    folds = cfg_raw.get("task", {}).get("folds", "all")
    folds_to_run = task.folds if folds == "all" else (folds if isinstance(folds, list) else [folds])

    # --- Build fold payloads once (with ids/y_true) ---
    fold_payload = {}
    all_structures = []
    for fold in folds_to_run:
        tr_x, tr_y = task.get_train_and_val_data(fold)
        te_x = task.get_test_data(fold, include_target=False)
        te_x_t, te_y = task.get_test_data(fold, include_target=True)
        ids = list(getattr(te_y, "index", range(len(te_y))))
        fold_payload[fold] = {
            "train_x": list(tr_x),
            "train_y": [float(v) for v in tr_y],
            "test_x": list(te_x),
            "test_y": [float(v) for v in te_y],
            "test_ids": [str(x) for x in ids],
        }
        all_structures.extend(list(tr_x))
        all_structures.extend(list(te_x))

    # --- Global graph cache (convert once, reuse all folds) ---
    cache_path = Path("data/mp_e_form_chgnet_graphs.pt")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    graph_cache: dict[str, Any] = {}
    converter_model = CHGNet.load()
    converter = converter_model.graph_converter

    if cache_path.exists():
        print(f"Loading graph cache: {cache_path}")
        graph_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        print("No graph cache found; creating global graph cache...")

    missing = 0
    adaptive_count = 0
    for i, s in enumerate(all_structures):
        k = structure_key(s)
        if k in graph_cache:
            continue
        g, used = convert_structure_with_adaptive_cutoff(s, converter, i)
        graph_cache[k] = g
        missing += 1
        if used is not None:
            adaptive_count += 1

    if missing > 0:
        print(f"Saving graph cache: {cache_path} (new={missing}, adaptive={adaptive_count})")
        torch.save(graph_cache, cache_path)

    out_dir = resolve_out_dir(cfg_raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_mae = {}
    for fold in folds_to_run:
        print(f"\n--- Starting Fold: {fold} ---")
        payload = fold_payload[fold]

        train_graphs = [graph_cache[structure_key(s)] for s in payload["train_x"]]
        test_graphs = [graph_cache[structure_key(s)] for s in payload["test_x"]]

        model = CHGNet.load()
        model.to(device)

        fold_tag = str(fold).replace("/", "_")
        history_csv = out_dir / f"history_fold_{fold_tag}.csv"
        ckpt_path = out_dir / f"model_fold_{fold_tag}.pth"

        train_one_fold(
            model,
            train_graphs,
            payload["train_y"],
            cfg,
            device,
            history_csv,
            ckpt_path,
        )

        preds = run_predict_on_graphs(model, test_graphs, device, cfg.batch_size)
        task.record(fold, preds)

        y_true = np.array(payload["test_y"], dtype=float)
        mae = float(np.mean(np.abs(preds - y_true)))
        fold_mae[fold] = mae
        print(f"Fold {fold} MAE: {mae:.6f}")

        preds_csv = out_dir / f"preds_fold_{fold_tag}.csv"
        with preds_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "y_true", "y_pred"])
            for sid, yt, yp in zip(payload["test_ids"], y_true.tolist(), preds.tolist()):
                w.writerow([sid, yt, yp])

    mean_mae = float(np.mean(list(fold_mae.values()))) if fold_mae else None

    result = {
        "task": {"name": task_name, "scores": task.scores},
        "metrics": {
            "metric_name": "MAE",
            "metric_value": mean_mae,
            "fold_scores": fold_mae,
            "mean": mean_mae,
            "std": float(np.std(list(fold_mae.values()))) if len(fold_mae) > 1 else 0.0,
        },
        "model": {
            "name": "CHGNet-Structure",
            "mode": cfg.mode,
            "freeze_backbone": cfg.freeze_backbone,
        },
        "hparams": train_dict,
        "runtime": {"device": str(device), "total_time_sec": time.perf_counter() - t0},
        "seed": seed,
        "graph_cache": {"path": str(cache_path), "size": len(graph_cache)},
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
