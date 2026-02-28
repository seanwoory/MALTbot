#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

try:
    import dgl
    from dgl.nn import GraphConv
except Exception:
    dgl = None
    GraphConv = None


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 128
    num_layers: int = 3
    cutoff: float = 8.0
    max_neighbors: int = 12
    data_fraction: float = 1.0
    cache_fraction: float = 0.1
    early_stopping_patience: int = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def structure_hash(s: Structure) -> str:
    return hashlib.sha1(json.dumps(s.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()


class AlignnCache:
    """Line-graph cache with retry and lightweight in-memory LRU for Drive I/O mitigation."""

    def __init__(self, cache_root: Path, mem_lru: int = 256):
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.mem_lru = mem_lru
        self._hot: dict[str, tuple[Any, Any]] = {}
        self._hot_order: list[str] = []

    def path_for(self, h: str) -> Path:
        return self.cache_root / f"{h}.bin"

    def has(self, h: str) -> bool:
        return self.path_for(h).exists()

    def _put_hot(self, h: str, g, lg):
        if h in self._hot:
            return
        self._hot[h] = (g, lg)
        self._hot_order.append(h)
        if len(self._hot_order) > self.mem_lru:
            old = self._hot_order.pop(0)
            self._hot.pop(old, None)

    def save(self, h: str, g, lg):
        dgl.save_graphs(str(self.path_for(h)), [g, lg])
        self._put_hot(h, g, lg)

    def load(self, h: str):
        if h in self._hot:
            return self._hot[h]
        gs, _ = dgl.load_graphs(str(self.path_for(h)))
        g, lg = gs[0], gs[1]
        self._put_hot(h, g, lg)
        return g, lg


def build_graph_with_retry(s: Structure, cutoff: float, max_neighbors: int, sid: str):
    for c in [cutoff, 10.0, 20.0]:
        try:
            return build_graphs(s, c, max_neighbors), None if c == cutoff else c
        except Exception as e:
            err = str(e).lower()
            if "isolated atom" in err or "no edges" in err:
                continue
            raise
    raise RuntimeError(f"graph build failed after retries sid={sid}")


def build_graphs(structure: Structure, cutoff: float, max_neighbors: int):
    n = len(structure)
    src, dst, dist = [], [], []
    for i in range(n):
        neigh = structure.get_neighbors(structure[i], cutoff)
        neigh = sorted(neigh, key=lambda x: x.nn_distance)[:max_neighbors]
        for nn in neigh:
            src.append(i)
            dst.append(int(nn.index))
            dist.append(float(nn.nn_distance))

    if len(src) == 0:
        raise ValueError("No edges found (isolated atom)")

    g = dgl.graph((src, dst), num_nodes=n)
    g.ndata["z"] = torch.tensor([int(site.specie.Z) for site in structure], dtype=torch.long)
    g.edata["d"] = torch.tensor(dist, dtype=torch.float32).view(-1, 1)

    lg = dgl.line_graph(g, backtracking=False, shared=True)
    lg.ndata["d"] = g.edata["d"] if g.num_edges() > 0 else torch.zeros((lg.num_nodes(), 1), dtype=torch.float32)
    return g, lg


class ALIGNNLike(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3, max_z=100):
        super().__init__()
        self.emb = nn.Embedding(max_z + 1, hidden_dim)
        self.eproj = nn.Linear(1, hidden_dim)
        self.g_layers = nn.ModuleList([GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True) for _ in range(num_layers)])
        self.l_layers = nn.ModuleList([GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, g, lg):
        h = self.emb(g.ndata["z"])
        e = self.eproj(lg.ndata["d"])
        for l in self.g_layers:
            h = torch.relu(l(g, h))
        for l in self.l_layers:
            e = torch.relu(l(lg, e))
        with g.local_scope(), lg.local_scope():
            g.ndata["h"] = h
            lg.ndata["e"] = e
            hg = dgl.mean_nodes(g, "h")
            he = dgl.mean_nodes(lg, "e")
        return self.head(torch.cat([hg, he], dim=-1)).view(-1)


def collate(batch):
    gs, lgs, ys = zip(*batch)
    return dgl.batch(gs), dgl.batch(lgs), torch.tensor(ys, dtype=torch.float32)


def run_preflight_lite(cfg: TrainConfig, folds_to_run, cache_root: Path, out_dir: Path, task, gh_push: bool):
    t0 = time.time()
    if dgl is None:
        raise RuntimeError("DGL not installed. install dgl first")

    print(
        "EFFECTIVE_CONFIG "
        f"folds={folds_to_run} epochs={cfg.epochs} batch={cfg.batch_size} "
        f"lr={cfg.lr} data_fraction={cfg.data_fraction} cache_fraction={cfg.cache_fraction}"
    )
    print(f"CUDA={torch.cuda.is_available()} cache_root={cache_root}")

    f0 = folds_to_run[0]
    tr_x, tr_y = task.get_train_and_val_data(f0)
    te_x = task.get_test_data(f0, include_target=False)
    print(f"sample_counts fold={f0} n_train_raw={len(tr_x)} n_test_raw={len(te_x)}")

    y = np.asarray(tr_y, dtype=float)
    q = np.quantile(y, [0.01, 0.5, 0.99]).tolist()
    print(f"target_scale_q01_q50_q99={q}")

    out_dir.mkdir(parents=True, exist_ok=True)
    t = out_dir / "preflight_alignn.json"
    t.write_text(json.dumps({"ok": True, "ts": time.time()}), encoding="utf-8")
    json.loads(t.read_text())
    t.unlink(missing_ok=True)

    if gh_push and not os.getenv("GH_TOKEN"):
        print("[WARN] GH_PUSH requested but GH_TOKEN missing")

    print(f"preflight_time_sec={time.time()-t0:.2f}")


def resolve_out_dir(cfg_raw: dict):
    date_str = cfg_raw.get("date") or datetime.now().astimezone().strftime("%Y-%m-%d")
    run_name = cfg_raw.get("run_name", "alignn_structure")
    out_root = Path(cfg_raw.get("output_root", "results/daily"))
    return out_root / date_str / run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preflight-lite", action="store_true")
    args = ap.parse_args()

    if dgl is None:
        raise RuntimeError("DGL import failed")

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg_raw.get("seed", 42))
    set_seed(seed)

    task_obj = cfg_raw.get("task", {})
    task_name = task_obj.get("name", "matbench_mp_e_form") if isinstance(task_obj, dict) else str(task_obj)
    params = cfg_raw.get("params", {})
    tr = cfg_raw.get("training", params)
    ex = cfg_raw.get("training_extras", params)

    cfg = TrainConfig(
        epochs=int(tr.get("epochs", 2)),
        batch_size=int(tr.get("batch_size", 8)),
        lr=float(tr.get("lr", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 1e-5)),
        hidden_dim=int(ex.get("hidden_dim", 128)),
        num_layers=int(ex.get("num_layers", 3)),
        cutoff=float(ex.get("cutoff", 8.0)),
        max_neighbors=int(ex.get("max_neighbors", 12)),
        data_fraction=float(ex.get("data_fraction", 1.0)),
        cache_fraction=float(ex.get("cache_fraction", ex.get("data_fraction", 1.0))),
        early_stopping_patience=int(ex.get("early_stopping_patience", 2)),
    )

    folds = task_obj.get("folds", "all") if isinstance(task_obj, dict) else "all"
    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    task = next(iter(mb.tasks))
    task.load()

    if folds == "all":
        folds_to_run = task.folds
    else:
        folds_to_run = folds if isinstance(folds, list) else [folds]
        norm = []
        for f in folds_to_run:
            if isinstance(f, str) and f.startswith("fold_"):
                try:
                    norm.append(int(f.split("_", 1)[1]))
                    continue
                except Exception:
                    pass
            norm.append(f)
        folds_to_run = norm

    drive_root = Path("/content/drive/MyDrive")
    cache_root = (drive_root / "MALTbot-cache" / "alignn_graph_cache" / task_name) if drive_root.exists() else (Path("data/alignn_graph_cache") / task_name)
    out_dir = resolve_out_dir(cfg_raw)

    if args.preflight_lite:
        run_preflight_lite(cfg, folds_to_run, cache_root, out_dir, task, gh_push=(os.getenv("MALTBOT_GH_PUSH") == "1"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = AlignnCache(cache_root)

    df = min(max(cfg.data_fraction, 0.0), 1.0)
    cf = min(max(cfg.cache_fraction, 0.0), 1.0)
    if cf < df:
        print(f"[WARN] cache_fraction({cf}) < data_fraction({df}); auto-promote cache_fraction={df}")
        cf = df

    fold_payload = {}
    cache_structs = []

    for fold in folds_to_run:
        tr_x, tr_y = task.get_train_and_val_data(fold)
        te_x = task.get_test_data(fold, include_target=False)
        _, te_y = task.get_test_data(fold, include_target=True)

        tr_x, tr_y = list(tr_x), [float(v) for v in tr_y]
        te_x, te_y = list(te_x), [float(v) for v in te_y]
        full_test_len = len(te_y)

        if df < 1.0:
            tr_keep = max(1, int(len(tr_x) * df)) if len(tr_x) > 0 else 0
            te_keep = max(1, int(len(te_x) * df)) if len(te_x) > 0 else 0
            if len(tr_x) > tr_keep:
                idx = np.random.choice(len(tr_x), tr_keep, replace=False)
                tr_x = [tr_x[i] for i in idx]
                tr_y = [tr_y[i] for i in idx]
            if len(te_x) > te_keep:
                idx = np.random.choice(len(te_x), te_keep, replace=False)
                te_x = [te_x[i] for i in idx]
                te_y = [te_y[i] for i in idx]

        c_tr_x, c_te_x = tr_x, te_x
        if cf < 1.0:
            c_tr_keep = max(1, int(len(tr_x) * cf)) if len(tr_x) > 0 else 0
            c_te_keep = max(1, int(len(te_x) * cf)) if len(te_x) > 0 else 0
            if len(tr_x) > c_tr_keep:
                idx = np.random.choice(len(tr_x), c_tr_keep, replace=False)
                c_tr_x = [tr_x[i] for i in idx]
            if len(te_x) > c_te_keep:
                idx = np.random.choice(len(te_x), c_te_keep, replace=False)
                c_te_x = [te_x[i] for i in idx]

        cache_structs.extend(c_tr_x)
        cache_structs.extend(c_te_x)

        fold_payload[fold] = {
            "train_x": tr_x,
            "train_y": tr_y,
            "test_x": te_x,
            "test_y": te_y,
            "full_test_len": full_test_len,
        }

    new_cache = 0
    failed_cache = 0
    for i, s in enumerate(cache_structs):
        h = structure_hash(s)
        if cache.has(h):
            continue
        try:
            (g, lg), retry_cutoff = build_graph_with_retry(s, cfg.cutoff, cfg.max_neighbors, h)
            if retry_cutoff is not None:
                print(f"[WARN] adaptive cutoff used sid={h[:10]} cutoff={retry_cutoff}")
            cache.save(h, g, lg)
            new_cache += 1
        except Exception as e:
            failed_cache += 1
            print(f"[WARN] cache graph failed idx={i} sid={h[:10]}: {e}")

    print(f"EFFECTIVE_CONFIG_EXT folds={folds_to_run} data_fraction={df} cache_fraction={cf} cache_root={cache_root} cache_new={new_cache} cache_failed={failed_cache}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_mae = {}

    for fold in folds_to_run:
        payload = fold_payload[fold]
        train = []
        for s, y in zip(payload["train_x"], payload["train_y"]):
            h = structure_hash(s)
            if cache.has(h):
                train.append((h, y))

        test = []
        for s, y in zip(payload["test_x"], payload["test_y"]):
            h = structure_hash(s)
            if cache.has(h):
                test.append((h, y))

        if len(train) == 0 or len(test) == 0:
            raise RuntimeError(f"fold={fold} no usable data after cache filtering")

        model = ALIGNNLike(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"EFFECTIVE_FOLD_DATA fold={fold} n_train={len(train)} n_test={len(test)}")
        print(f"EFFECTIVE_PARAM_COUNT total={total_params} trainable={trainable_params}")

        idx = np.random.permutation(len(train))
        val_n = max(1, int(0.1 * len(idx)))
        val_idx = idx[:val_n]
        tr_idx = idx[val_n:] if len(idx[val_n:]) > 0 else idx[:val_n]

        tr_set = [train[i] for i in tr_idx]
        va_set = [train[i] for i in val_idx]

        opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        crit = nn.HuberLoss()

        best = float("inf")
        no_imp = 0
        hist_rows = []
        ckpt = out_dir / f"model_fold_{fold}.pth"
        hist = out_dir / f"history_fold_{fold}.csv"

        for ep in range(1, cfg.epochs + 1):
            t_ep = time.time()
            model.train()
            losses = []
            random.shuffle(tr_set)
            for i in range(0, len(tr_set), cfg.batch_size):
                b = tr_set[i:i+cfg.batch_size]
                gs, lgs, ys = [], [], []
                for h, y in b:
                    g, lg = cache.load(h)
                    gs.append(g)
                    lgs.append(lg)
                    ys.append(y)
                bg = dgl.batch(gs).to(device)
                blg = dgl.batch(lgs).to(device)
                y = torch.tensor(ys, dtype=torch.float32, device=device)
                opt.zero_grad()
                pred = model(bg, blg)
                loss = crit(pred, y)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            model.eval()
            vpred, vtrue = [], []
            with torch.no_grad():
                for i in range(0, len(va_set), cfg.batch_size):
                    b = va_set[i:i+cfg.batch_size]
                    gs, lgs, ys = [], [], []
                    for h, y in b:
                        g, lg = cache.load(h)
                        gs.append(g)
                        lgs.append(lg)
                        ys.append(y)
                    bg = dgl.batch(gs).to(device)
                    blg = dgl.batch(lgs).to(device)
                    p = model(bg, blg).detach().cpu().numpy().tolist()
                    vpred.extend(p)
                    vtrue.extend(ys)

            val_mae = float(np.mean(np.abs(np.array(vpred) - np.array(vtrue))))
            train_loss = float(np.mean(losses)) if losses else float("nan")
            hist_rows.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_mae, "val_mae": val_mae})
            eta = ((time.time() - t_ep) * (cfg.epochs - ep)) / 60.0
            print(f"Epoch {ep}/{cfg.epochs} - Loss: {train_loss:.6f} - Val MAE: {val_mae:.6f} - ETA: {eta:.1f}m")

            if val_mae < best:
                best = val_mae
                no_imp = 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_imp += 1
                if no_imp >= cfg.early_stopping_patience:
                    break

        with hist.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae"])
            w.writeheader()
            w.writerows(hist_rows)

        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))

        model.eval()
        preds, y_true = [], []
        with torch.no_grad():
            for i in range(0, len(test), cfg.batch_size):
                b = test[i:i+cfg.batch_size]
                gs, lgs, ys = [], [], []
                for h, y in b:
                    g, lg = cache.load(h)
                    gs.append(g)
                    lgs.append(lg)
                    ys.append(y)
                bg = dgl.batch(gs).to(device)
                blg = dgl.batch(lgs).to(device)
                p = model(bg, blg).detach().cpu().numpy().tolist()
                preds.extend(p)
                y_true.extend(ys)

        preds = np.array(preds, dtype=float)
        y_true = np.array(y_true, dtype=float)
        mae = float(np.mean(np.abs(preds - y_true)))
        fold_mae[fold] = mae

        matbench_recorded = (df >= 1.0 and cf >= 1.0 and len(preds) == payload["full_test_len"])
        if matbench_recorded:
            task.record(fold, preds)
        else:
            print(f"[WARN] skip task.record fold={fold} len(preds)={len(preds)} full_test_len={payload['full_test_len']}")

        preds_csv = out_dir / f"preds_fold_{fold}.csv"
        with preds_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "y_true", "y_pred"])
            for j, (yt, yp) in enumerate(zip(y_true.tolist(), preds.tolist())):
                w.writerow([j, yt, yp])

    mean_mae = float(np.mean(list(fold_mae.values()))) if fold_mae else None

    result = {
        "task": {"name": task_name, "scores": task.scores if (df >= 1.0 and cf >= 1.0) else None},
        "metrics": {
            "metric_name": "MAE",
            "metric_value": mean_mae,
            "fold_scores": fold_mae,
            "mean": mean_mae,
            "std": float(np.std(list(fold_mae.values()))) if len(fold_mae) > 1 else 0.0,
        },
        "model": {"name": "ALIGNN-like", "framework": "DGL"},
        "hparams": tr,
        "effective_params": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "cutoff": cfg.cutoff,
            "max_neighbors": cfg.max_neighbors,
            "data_fraction": df,
            "cache_fraction": cf,
            "early_stopping_patience": cfg.early_stopping_patience,
            "folds": folds_to_run,
        },
        "runtime": {"device": str(device), "total_time_sec": time.time() - t0},
        "seed": seed,
    }

    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote results to {out_file}")
    if mean_mae is not None:
        print(f"FINAL_METRIC_MAE={mean_mae:.6f}")


if __name__ == "__main__":
    main()
