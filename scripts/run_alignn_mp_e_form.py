#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Structure

try:
    import dgl
    import dgl.function as fn
except Exception:
    dgl = None


class EdgeGatedGraphConv(nn.Module):
    """ALIGNN-style edge-gated graph convolution for atom/bond graph."""

    def __init__(self, dim: int, edge_dim: int | None = None):
        super().__init__()
        edge_dim = edge_dim or dim
        self.src_proj = nn.Linear(dim, dim)
        self.dst_proj = nn.Linear(dim, dim)
        self.edge_gate = nn.Linear(edge_dim, dim)
        self.msg_proj = nn.Linear(dim, dim)
        self.node_upd = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.edge_upd = nn.Sequential(
            nn.Linear(edge_dim + 2 * dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, g, h, e):
        with g.local_scope():
            g.ndata["h"] = h
            g.edata["e"] = e

            g.apply_edges(lambda edges: {
                "gate_logits": self.src_proj(edges.src["h"]) + self.dst_proj(edges.dst["h"]) + self.edge_gate(edges.data["e"]),
                "src_msg": self.msg_proj(edges.src["h"]),
            })
            gate = torch.sigmoid(g.edata["gate_logits"])
            g.edata["m"] = gate * g.edata["src_msg"]
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "agg"))

            h_new = self.node_norm(h + self.node_upd(torch.cat([h, g.ndata["agg"]], dim=-1)))

            e_input = torch.cat([e, g.edata["src_msg"], self.msg_proj(g.dstdata["h"][g.edges()[1]])], dim=-1)
            e_new = self.edge_norm(e + self.edge_upd(e_input))

        return h_new, e_new


class AngleGatedConv(nn.Module):
    """Message passing on line-graph with angle-aware gating."""

    def __init__(self, dim: int):
        super().__init__()
        self.src_proj = nn.Linear(dim, dim)
        self.dst_proj = nn.Linear(dim, dim)
        self.angle_proj = nn.Linear(dim, dim)
        self.msg_proj = nn.Linear(dim, dim)
        self.upd = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, lg, e, a):
        with lg.local_scope():
            lg.ndata["e"] = e
            lg.edata["a"] = a
            lg.apply_edges(lambda edges: {
                "gate_logits": self.src_proj(edges.src["e"]) + self.dst_proj(edges.dst["e"]) + self.angle_proj(edges.data["a"]),
                "src_msg": self.msg_proj(edges.src["e"]),
            })
            gate = torch.sigmoid(lg.edata["gate_logits"])
            lg.edata["m"] = gate * lg.edata["src_msg"]
            lg.update_all(fn.copy_e("m", "m"), fn.sum("m", "agg"))
            e_new = self.norm(e + self.upd(torch.cat([e, lg.ndata["agg"]], dim=-1)))
        return e_new


class ALIGNNLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.angle_conv = AngleGatedConv(dim)
        self.edge_atom_conv = EdgeGatedGraphConv(dim)

    def forward(self, g, lg, h, e, a):
        e = self.angle_conv(lg, e, a)
        h, e = self.edge_atom_conv(g, h, e)
        return h, e


class ALIGNN(nn.Module):
    """SOTA-aligned core blocks: angle MP + edge-gated conv."""

    def __init__(self, hidden_dim=256, num_layers=4, max_z=100):
        super().__init__()
        self.atom_emb = nn.Embedding(max_z + 1, hidden_dim)
        self.edge_emb = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.angle_emb = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([ALIGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs):
        g, lg = inputs
        h = self.atom_emb(g.ndata["z"])
        e = self.edge_emb(g.edata["d"])
        a = self.angle_emb(lg.edata["angle"])

        for layer in self.layers:
            h, e = layer(g, lg, h, e, a)

        with g.local_scope():
            g.ndata["h"] = h
            hg = dgl.mean_nodes(g, "h")
        return self.readout(hg).view(-1)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 256
    num_layers: int = 4
    cutoff: float = 8.0
    max_neighbors: int = 12
    data_fraction: float = 1.0
    cache_fraction: float = 1.0
    early_stopping_patience: int = 5
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.05


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def structure_hash(s: Structure) -> str:
    import hashlib

    key = json.dumps(s.as_dict(), sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _safe_angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    dot = (u * v).sum(dim=-1)
    nu = torch.norm(u, dim=-1)
    nv = torch.norm(v, dim=-1)
    cos = dot / (nu * nv + 1e-8)
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos).unsqueeze(-1)


def build_dgl_graph(structure: Structure, cutoff: float, max_neighbors: int):
    n = len(structure)
    src, dst, dist, vec = [], [], [], []
    for i in range(n):
        neigh = structure.get_neighbors(structure[i], cutoff)
        neigh = sorted(neigh, key=lambda x: x.nn_distance)[:max_neighbors]
        for nn_site in neigh:
            src.append(i)
            dst.append(int(nn_site.index))
            d = float(nn_site.nn_distance)
            dist.append(d)
            v = np.array(nn_site.coords) - np.array(structure[i].coords)
            vec.append(v.astype(np.float32))

    if len(src) == 0:
        raise ValueError("No edges found in structure graph")

    g = dgl.graph((src, dst), num_nodes=n)
    g.ndata["z"] = torch.tensor([int(site.specie.Z) for site in structure], dtype=torch.long)
    g.edata["d"] = torch.tensor(dist, dtype=torch.float32).view(-1, 1)
    g.edata["vec"] = torch.tensor(np.stack(vec), dtype=torch.float32)

    lg = dgl.line_graph(g, backtracking=False, shared=True)

    # Line-graph edges represent bond pairs (i->j, j->k): angle at j between (-vec_ij) and vec_jk.
    lu, lv = lg.edges()
    v1 = -g.edata["vec"][lu]
    v2 = g.edata["vec"][lv]
    lg.edata["angle"] = _safe_angle(v1, v2)

    return g, lg


def build_dgl_graph_with_adaptive_cutoff(structure: Structure, cutoff: float, max_neighbors: int):
    last_err = None
    for c in [cutoff, max(cutoff, 10.0), max(cutoff, 20.0)]:
        try:
            return build_dgl_graph(structure, c, max_neighbors), c
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Graph build failed after adaptive cutoffs: {last_err}")


class GraphDiskCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, h: str) -> Path:
        return self.root / f"{h}.bin"

    def has(self, h: str) -> bool:
        return self.path_for(h).exists()

    def put(self, h: str, g, lg):
        dgl.save_graphs(str(self.path_for(h)), [g, lg])

    def get(self, h: str):
        gs, _ = dgl.load_graphs(str(self.path_for(h)))
        return gs[0], gs[1]


def preflight_lite(cfg: TrainConfig, folds_to_run, cache_root: Path, out_dir: Path, task):
    print(
        "EFFECTIVE_CONFIG "
        f"folds={folds_to_run} epochs={cfg.epochs} batch={cfg.batch_size} lr={cfg.lr} "
        f"hidden={cfg.hidden_dim} layers={cfg.num_layers} cutoff={cfg.cutoff} max_neighbors={cfg.max_neighbors} "
        f"data_fraction={cfg.data_fraction} cache_fraction={cfg.cache_fraction} patience={cfg.early_stopping_patience} "
        f"warmup_epochs={cfg.warmup_epochs} min_lr_ratio={cfg.min_lr_ratio}"
    )
    if dgl is None:
        raise RuntimeError("DGL missing")
    print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
    print(f"[cache] cache_root={cache_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] out_dir={out_dir}")
    tr_x, tr_y = task.get_train_and_val_data(folds_to_run[0])
    print(f"EFFECTIVE_SPLIT fold={folds_to_run[0]} n_train={len(tr_x)}")
    print(f"target_scale_check={np.mean(tr_y):.4f} eV/atom")


def resolve_out_dir(cfg_raw: dict):
    date_str = cfg_raw.get("date") or datetime.now().strftime("%Y-%m-%d")
    run_name = cfg_raw.get("run_name", "alignn_mp_e_form")
    return Path(cfg_raw.get("output_root", "results/daily")) / date_str / run_name


def make_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    warmup_steps = max(1, min(warmup_steps, max(1, total_steps - 1)))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preflight-lite", action="store_true")
    args = ap.parse_args()

    if dgl is None:
        raise RuntimeError("DGL import failed.")

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg_raw.get("seed", 42)))

    params = cfg_raw.get("params", {})
    tr = cfg_raw.get("training", params)
    ex = cfg_raw.get("training_extras", params)

    cfg = TrainConfig(
        epochs=int(tr.get("epochs", 30)),
        batch_size=int(tr.get("batch_size", 64)),
        lr=float(tr.get("lr", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 1e-5)),
        hidden_dim=int(ex.get("hidden_dim", 256)),
        num_layers=int(ex.get("num_layers", 4)),
        cutoff=float(ex.get("cutoff", 8.0)),
        max_neighbors=int(ex.get("max_neighbors", 12)),
        data_fraction=float(ex.get("data_fraction", 1.0)),
        cache_fraction=float(ex.get("cache_fraction", ex.get("data_fraction", 1.0))),
        early_stopping_patience=int(ex.get("early_stopping_patience", 5)),
        warmup_epochs=int(ex.get("warmup_epochs", 1)),
        min_lr_ratio=float(ex.get("min_lr_ratio", 0.05)),
    )

    task_name = cfg_raw.get("task", {}).get("name", "matbench_mp_e_form")
    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    task = next(iter(mb.tasks))
    task.load()

    folds = cfg_raw.get("task", {}).get("folds", "all")
    folds_to_run = task.folds if folds == "all" else (folds if isinstance(folds, list) else [folds])
    norm_folds = []
    for f in folds_to_run:
        if isinstance(f, str) and f.startswith("fold_"):
            norm_folds.append(int(f.split("_")[1]))
        else:
            norm_folds.append(int(f))
    folds_to_run = norm_folds

    drive_root = Path("/content/drive/MyDrive")
    cache_root = (drive_root / "MALTbot-cache" / "alignn_graph_cache" / task_name) if drive_root.exists() else (Path("data/alignn_graph_cache") / task_name)
    out_dir = resolve_out_dir(cfg_raw)

    if args.preflight_lite:
        preflight_lite(cfg, folds_to_run, cache_root, out_dir, task)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = GraphDiskCache(cache_root)

    fold_payload = {}
    all_structs = []
    for fold in folds_to_run:
        tr_x, tr_y = task.get_train_and_val_data(fold)
        te_x, te_y = task.get_test_data(fold, include_target=True)

        if cfg.data_fraction < 1.0:
            tr_keep = max(1, int(len(tr_x) * cfg.data_fraction))
            te_keep = max(1, int(len(te_x) * cfg.data_fraction))
            tr_idx = np.random.choice(len(tr_x), tr_keep, replace=False)
            te_idx = np.random.choice(len(te_x), te_keep, replace=False)
            tr_x, tr_y = [tr_x[i] for i in tr_idx], [tr_y[i] for i in tr_idx]
            te_x, te_y = [te_x[i] for i in te_idx], [te_y[i] for i in te_idx]

        fold_payload[fold] = {
            "tr_x": tr_x,
            "tr_y": list(tr_y),
            "te_x": te_x,
            "te_y": list(te_y),
            "full_test_len": len(task.get_test_data(fold, include_target=False)),
        }
        all_structs.extend(tr_x)
        all_structs.extend(te_x)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_hashes = [structure_hash(s) for s in all_structs]
    unique_hashes = list(dict.fromkeys(all_hashes))
    n_cache_target = max(1, int(len(unique_hashes) * cfg.cache_fraction)) if cfg.cache_fraction < 1.0 else len(unique_hashes)
    cache_targets = set(unique_hashes[:n_cache_target])

    built, skipped = 0, 0
    for s in all_structs:
        h = structure_hash(s)
        if h not in cache_targets:
            continue
        if not cache.has(h):
            try:
                (g, lg), _ = build_dgl_graph_with_adaptive_cutoff(s, cfg.cutoff, cfg.max_neighbors)
                cache.put(h, g, lg)
                built += 1
            except Exception:
                skipped += 1
    print(f"[cache] built={built} skipped={skipped} target={len(cache_targets)}")

    fold_mae = {}
    for fold in folds_to_run:
        print(f"\n--- Starting Fold: {fold} ---")
        model = ALIGNN(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers).to(device)
        p = fold_payload[fold]

        train_data = []
        for s, y in zip(p["tr_x"], p["tr_y"]):
            h = structure_hash(s)
            if cache.has(h):
                train_data.append((h, y))

        if len(train_data) < 2:
            raise RuntimeError(f"Not enough cached train samples for fold={fold}: {len(train_data)}")

        idx = np.random.permutation(len(train_data))
        val_n = max(1, int(0.1 * len(idx)))
        tr_set = [train_data[i] for i in idx[val_n:]]
        va_set = [train_data[i] for i in idx[:val_n]]
        if len(tr_set) == 0:
            tr_set = va_set

        opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        steps_per_epoch = max(1, math.ceil(len(tr_set) / cfg.batch_size))
        total_steps = max(1, steps_per_epoch * cfg.epochs)
        warmup_steps = cfg.warmup_epochs * steps_per_epoch
        sched = make_warmup_cosine_scheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=cfg.min_lr_ratio)

        crit = nn.L1Loss()
        best_mae = float("inf")
        best_ep = -1
        global_step = 0

        for ep in range(1, cfg.epochs + 1):
            model.train()
            losses = []
            for i in range(0, len(tr_set), cfg.batch_size):
                batch = tr_set[i : i + cfg.batch_size]
                gs, lgs, ys = [], [], []
                for h, y in batch:
                    g, lg = cache.get(h)
                    gs.append(g)
                    lgs.append(lg)
                    ys.append(y)
                bg, blg = dgl.batch(gs).to(device), dgl.batch(lgs).to(device)
                y = torch.tensor(ys, dtype=torch.float32, device=device)

                opt.zero_grad()
                pred = model([bg, blg])
                loss = crit(pred, y)
                loss.backward()
                opt.step()
                sched.step()
                global_step += 1
                losses.append(loss.item())

            model.eval()
            v_mae = []
            with torch.no_grad():
                for i in range(0, len(va_set), cfg.batch_size):
                    batch = va_set[i : i + cfg.batch_size]
                    gs, lgs, ys = [], [], []
                    for h, y in batch:
                        g, lg = cache.get(h)
                        gs.append(g)
                        lgs.append(lg)
                        ys.append(y)
                    bg, blg = dgl.batch(gs).to(device), dgl.batch(lgs).to(device)
                    pred = model([bg, blg])
                    v_mae.extend(torch.abs(pred - torch.tensor(ys, dtype=torch.float32, device=device)).cpu().numpy())

            cur_v_mae = float(np.mean(v_mae)) if v_mae else float("inf")
            cur_lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep}/{cfg.epochs} - Step {global_step}/{total_steps} - LR: {cur_lr:.6e} - Loss: {np.mean(losses):.4f} - Val MAE: {cur_v_mae:.4f}")

            if cur_v_mae < best_mae:
                best_mae = cur_v_mae
                best_ep = ep
                torch.save(model.state_dict(), out_dir / f"model_fold_{fold}.pth")
            elif (ep - best_ep) >= cfg.early_stopping_patience:
                print(f"[early-stop] fold={fold} ep={ep} best_ep={best_ep} best_val_mae={best_mae:.4f}")
                break

        model.load_state_dict(torch.load(out_dir / f"model_fold_{fold}.pth", map_location=device))
        model.eval()
        t_mae = []
        with torch.no_grad():
            for s, y in zip(p["te_x"], p["te_y"]):
                h = structure_hash(s)
                if cache.has(h):
                    g, lg = cache.get(h)
                    pred = model([dgl.batch([g]).to(device), dgl.batch([lg]).to(device)])
                    t_mae.append(abs(pred.item() - y))

        fold_score = float(np.mean(t_mae)) if t_mae else float("inf")
        fold_mae[str(fold)] = fold_score
        print(f"Fold {fold} Test MAE: {fold_score:.4f}")

    metric = float(np.mean(list(fold_mae.values()))) if fold_mae else float("inf")
    res = {
        "status": "success",
        "experiment": "alignn_fold0_agile",
        "metric_name": "MAE",
        "metric_value": metric,
        "metrics": {
            "metric_name": "MAE",
            "metric_unit": "eV/atom",
            "fold_scores": fold_mae,
            "mean": metric,
            "std": float(np.std(list(fold_mae.values()))) if len(fold_mae) > 1 else 0.0,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"FINAL_METRIC_MAE={metric:.6f}")


if __name__ == "__main__":
    main()
