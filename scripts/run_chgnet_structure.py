#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
    train_fraction: float = 1.0
    val_fraction: float = 1.0
    early_stopping_patience: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def structure_key(s: Structure) -> str:
    return json.dumps(s.as_dict(), sort_keys=True, ensure_ascii=False)


def structure_hash(s: Structure) -> str:
    return hashlib.sha1(structure_key(s).encode("utf-8")).hexdigest()


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

    raise RuntimeError(f"Graph conversion failed for structure idx={idx} after adaptive cutoffs (10.0, 20.0).")


class ChunkedGraphCache:
    """Disk-backed graph cache with chunked .pt files (e.g., 1000 graphs/file)."""

    def __init__(self, cache_dir: Path, chunk_size: int = 1000):
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.chunks_dir = self.cache_dir / "chunks"
        self.manifest_path = self.cache_dir / "manifest.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self.key_to_ref: dict[str, tuple[str, int]] = {}
        self._current_chunk_file: str | None = None
        self._current_chunk_graphs: list[Any] | None = None

        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.key_to_ref = {k: (v[0], int(v[1])) for k, v in raw.get("key_to_ref", {}).items()}

    def _save_manifest(self):
        payload = {
            "chunk_size": self.chunk_size,
            "key_to_ref": self.key_to_ref,
        }
        self.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    def _next_chunk_index(self) -> int:
        files = sorted(self.chunks_dir.glob("chunk_*.pt"))
        if not files:
            return 0
        last = files[-1].stem.split("_")[-1]
        return int(last) + 1

    def build_missing(self, structures: list[Structure], converter) -> tuple[int, int]:
        """Convert only missing structures and append to chunk files."""
        buffer_graphs: list[Any] = []
        buffer_keys: list[str] = []
        new_count = 0
        adaptive_count = 0
        chunk_idx = self._next_chunk_index()

        def flush_buffer():
            nonlocal chunk_idx, buffer_graphs, buffer_keys
            if not buffer_graphs:
                return
            chunk_file = f"chunk_{chunk_idx:05d}.pt"
            out = self.chunks_dir / chunk_file
            torch.save({"keys": buffer_keys, "graphs": buffer_graphs}, out)
            for i, k in enumerate(buffer_keys):
                self.key_to_ref[k] = (chunk_file, i)
            chunk_idx += 1
            buffer_graphs = []
            buffer_keys = []

        for i, s in enumerate(structures):
            k = structure_hash(s)
            if k in self.key_to_ref:
                continue
            g, used = convert_structure_with_adaptive_cutoff(s, converter, i)
            buffer_graphs.append(g)
            buffer_keys.append(k)
            new_count += 1
            if used is not None:
                adaptive_count += 1
            if len(buffer_graphs) >= self.chunk_size:
                flush_buffer()

        flush_buffer()
        if new_count > 0:
            self._save_manifest()
        return new_count, adaptive_count

    def ref_for_structure(self, s: Structure) -> tuple[str, int]:
        k = structure_hash(s)
        if k not in self.key_to_ref:
            raise KeyError("Structure missing from chunk cache")
        return self.key_to_ref[k]

    def load_graph(self, ref: tuple[str, int]):
        chunk_file, offset = ref
        if self._current_chunk_file != chunk_file or self._current_chunk_graphs is None:
            payload = torch.load(self.chunks_dir / chunk_file, map_location="cpu", weights_only=False)
            self._current_chunk_file = chunk_file
            self._current_chunk_graphs = payload["graphs"]
        return self._current_chunk_graphs[offset]


def move_graph_batch_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except Exception:
            pass
    if isinstance(obj, list):
        return [move_graph_batch_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_graph_batch_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_graph_batch_to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        try:
            for k, v in obj.__dict__.items():
                setattr(obj, k, move_graph_batch_to_device(v, device))
        except Exception:
            pass
    return obj


class GraphChunkDataset(Dataset):
    def __init__(self, refs: list[tuple[str, int]], targets: list[float], cache: ChunkedGraphCache):
        self.refs = refs
        self.targets = targets
        self.cache = cache

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        g = self.cache.load_graph(self.refs[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return g, y


def collate_or_fallback(batch):
    graphs, targets = zip(*batch)
    try:
        from chgnet.data.loader import collate_graphs

        return collate_graphs(list(graphs)), torch.stack(targets)
    except Exception:
        return list(graphs), torch.stack(targets)


def run_predict_on_refs(model, refs: list[tuple[str, int]], cache: ChunkedGraphCache, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds: list[float] = []
    with torch.no_grad():
        for i in range(0, len(refs), batch_size):
            chunk_refs = refs[i : i + batch_size]
            chunk_graphs = [cache.load_graph(r) for r in chunk_refs]

            if hasattr(model, "predict_graph"):
                chunk_graphs = move_graph_batch_to_device(chunk_graphs, device)
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
                bg = move_graph_batch_to_device(bg, device)
                out = model(bg, task="e")
                pe = out["e"] if isinstance(out, dict) else out
                preds.extend([float(x) for x in np.array(pe.detach().cpu()).reshape(-1)])
    return np.array(preds)


def train_one_fold(
    model,
    train_refs: list[tuple[str, int]],
    train_targets: list[float],
    cache: ChunkedGraphCache,
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

    n = len(train_refs)
    idx = np.random.permutation(n)
    val_n = max(1, int(0.1 * n))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:] if len(idx[val_n:]) > 0 else idx[:val_n]

    tr_refs = [train_refs[i] for i in tr_idx]
    tr_targets = [train_targets[i] for i in tr_idx]
    va_refs = [train_refs[i] for i in val_idx]
    va_targets = [train_targets[i] for i in val_idx]

    # Agile mode: fractional dataset sampling
    tf = min(max(cfg.train_fraction, 0.0), 1.0)
    vf = min(max(cfg.val_fraction, 0.0), 1.0)
    if tf < 1.0 and len(tr_refs) > 1:
        keep = max(1, int(len(tr_refs) * tf))
        pick = np.random.choice(len(tr_refs), size=keep, replace=False)
        tr_refs = [tr_refs[i] for i in pick]
        tr_targets = [tr_targets[i] for i in pick]
    if vf < 1.0 and len(va_refs) > 1:
        keep = max(1, int(len(va_refs) * vf))
        pick = np.random.choice(len(va_refs), size=keep, replace=False)
        va_refs = [va_refs[i] for i in pick]
        va_targets = [va_targets[i] for i in pick]

    train_loader = DataLoader(
        GraphChunkDataset(tr_refs, tr_targets, cache),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_or_fallback,
        num_workers=0,
    )

    criterion = nn.HuberLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_mae = float("inf")
    rows = []
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    no_improve_epochs = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for bg, y in train_loader:
            bg = move_graph_batch_to_device(bg, device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(bg, task="e")
            pe = out["e"] if isinstance(out, dict) else out
            loss = criterion(pe.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else None
        val_pred = run_predict_on_refs(model, va_refs, cache, device, cfg.batch_size)
        val_true = np.array(va_targets, dtype=float)
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
        val_loss = float(np.mean(np.abs(val_pred - val_true)))

        rows.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae})
        print(f"Epoch {ep}/{cfg.epochs} train_loss={train_loss:.6f} val_mae={val_mae:.6f}")

        if val_mae < best_mae:
            best_mae = val_mae
            no_improve_epochs = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= max(1, cfg.early_stopping_patience):
                print(
                    f"Early stopping at epoch {ep}: no val MAE improvement for "
                    f"{cfg.early_stopping_patience} epochs"
                )
                break

    with history_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae"])
        w.writeheader()
        w.writerows(rows)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))


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
        train_fraction=float(extras.get("train_fraction", extras.get("data_fraction", 1.0))),
        val_fraction=float(extras.get("val_fraction", extras.get("data_fraction", 1.0))),
        early_stopping_patience=int(extras.get("early_stopping_patience", 5)),
    )

    device_cfg = cfg_raw.get("runtime", {}).get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and device_cfg == "auto" else "cpu")
    print(f"Using device: {device}")

    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    task = next(iter(mb.tasks))
    task.load()

    folds = cfg_raw.get("task", {}).get("folds", "all")
    if folds == "all":
        folds_to_run = task.folds
    else:
        folds_to_run = folds if isinstance(folds, list) else [folds]
        normalized = []
        for f in folds_to_run:
            if isinstance(f, str) and f.startswith("fold_"):
                try:
                    normalized.append(int(f.split("_", 1)[1]))
                    continue
                except Exception:
                    pass
            normalized.append(f)
        folds_to_run = normalized

    chunk_size = int(extras.get("graph_cache_chunk_size", 1000))
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        cache_root = drive_root / "MALTbot-cache" / "chgnet_graph_cache"
    else:
        cache_root = Path("data/chgnet_graph_cache")
    graph_cache = ChunkedGraphCache(cache_root / task_name, chunk_size=chunk_size)

    converter_model = CHGNet.load()
    converter = converter_model.graph_converter

    fold_payload = {}
    all_structs = []
    for fold in folds_to_run:
        tr_x, tr_y = task.get_train_and_val_data(fold)
        te_x = task.get_test_data(fold, include_target=False)
        _, te_y = task.get_test_data(fold, include_target=True)
        ids = list(getattr(te_y, "index", range(len(te_y))))

        fold_payload[fold] = {
            "train_x": list(tr_x),
            "train_y": [float(v) for v in tr_y],
            "test_x": list(te_x),
            "test_y": [float(v) for v in te_y],
            "test_ids": [str(x) for x in ids],
        }
        all_structs.extend(list(tr_x))
        all_structs.extend(list(te_x))

    print(f"Building/updating chunked graph cache (chunk_size={chunk_size}) ...")
    new_count, adaptive_count = graph_cache.build_missing(all_structs, converter)
    print(f"Chunked graph cache ready: {graph_cache.cache_dir} (new={new_count}, adaptive={adaptive_count})")

    out_dir = resolve_out_dir(cfg_raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_mae = {}
    for fold in folds_to_run:
        print(f"\n--- Starting Fold: {fold} ---")
        payload = fold_payload[fold]

        train_refs = [graph_cache.ref_for_structure(s) for s in payload["train_x"]]
        test_refs = [graph_cache.ref_for_structure(s) for s in payload["test_x"]]

        model = CHGNet.load()
        model.to(device)

        fold_tag = str(fold).replace("/", "_")
        history_csv = out_dir / f"history_fold_{fold_tag}.csv"
        ckpt_path = out_dir / f"model_fold_{fold_tag}.pth"

        train_one_fold(
            model,
            train_refs,
            payload["train_y"],
            graph_cache,
            cfg,
            device,
            history_csv,
            ckpt_path,
        )

        preds = run_predict_on_refs(model, test_refs, graph_cache, device, cfg.batch_size)
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
        "graph_cache": {
            "dir": str(graph_cache.cache_dir),
            "chunk_size": chunk_size,
            "num_cached_structures": len(graph_cache.key_to_ref),
        },
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
