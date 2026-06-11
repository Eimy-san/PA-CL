"""CLI entry point: run a single experiment cell from a YAML config.

Usage:
    python scripts/run_experiment.py --config configs/poc_cpmnist200.yaml \
        --method pacl --seed 0

Artifacts written to <out_dir>/<method>/seed<seed>/:
    summary.json     ACC, BWT, plasticity_decay (final-row + diag protocol),
                     plus per-task forgetting, wall-clock seconds, etc.
    diag_acc.npy     length-T array, plasticity-decay raw signal
    final_row.npy    length-T array, model's final acc on each task
    rank_traj.json   length-T list of {layer: erank} dicts
    rank_floor_traj.json (PA-CL only) length-T list of {layer: floor}
    train_loss.npy   length-T array, mean task-loss per task
    config.yaml      copy of the resolved config actually used
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Make pa_cl importable when running from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pa_cl.datasets import build_dataset
from pa_cl.models import build_model
from pa_cl.baselines import build_baseline
from pa_cl.pacl import PACL, PACLConfig
from pa_cl.trainer import train_continual
from pa_cl.metrics import (
    acc_sparse, bwt_sparse, plasticity_decay, per_task_forgetting_sparse
)


METHOD_REGISTRY = {
    # method_name -> (baseline_name, use_pacl)
    "erm":         ("erm",   False),
    "cbp":         ("cbp",   False),
    "er":          ("er",    False),
    "derpp":       ("derpp", False),
    "ewc":         ("ewc",   False),
    "agem":        ("agem",  False),
    "pacl":        ("erm",   True),    # PA-CL on top of ERM by default
    "pacl_erm":    ("erm",   True),
    "pacl_cbp":    ("cbp",   True),
    "pacl_er":     ("er",    True),
    "pacl_derpp":  ("derpp", True),
    "pacl_ewc":    ("ewc",   True),
    "pacl_agem":   ("agem",  True),
}


def resolve_method(cfg: dict, method: str) -> dict:
    """Mutate cfg in place to set baseline / pacl.enabled for the method."""
    base_name, use_pacl = METHOD_REGISTRY[method]
    cfg = copy.deepcopy(cfg)
    cfg["baseline"] = base_name
    cfg.setdefault("pacl", {})
    cfg["pacl"]["enabled"] = bool(use_pacl)
    cfg["_method"] = method
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--method", default=None,
                    help="One of erm, cbp, pacl, pacl_erm, pacl_cbp.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override seed from config.")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Override out_dir from config.")
    ap.add_argument("--lam", type=float, default=None,
                    help="Override PA-CL lambda from config (for sweeps).")
    ap.add_argument("--disable_projection", action="store_true",
                    help="Disable the Fisher-null projection (Sec 6.2 "
                         "ablation: rank gradient applied isotropically).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Override the method/leaf subdir name "
                         "(useful when sweeping a single method across "
                         "hyperparameters that all collapse to the same "
                         "method registry entry).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.method is not None:
        cfg = resolve_method(cfg, args.method)
    method = cfg.get("_method", "custom")
    if args.tag is not None:
        method = args.tag
        cfg["_method"] = method

    if args.lam is not None:
        cfg.setdefault("pacl", {})["lam"] = float(args.lam)
    if args.disable_projection:
        cfg.setdefault("pacl", {})["disable_projection"] = True

    if args.seed is not None:
        cfg["dataset"]["seed"] = args.seed
    seed = cfg["dataset"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_out = Path(args.out_dir or cfg.get("out_dir", "runbook/_default"))
    out_dir = base_out / method / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device.
    dev_req = cfg["training"].get("device", "cuda")
    device = torch.device(
        dev_req if (dev_req == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[run] method={method} seed={seed} device={device}")

    # Data.
    train_loaders, test_loaders = build_dataset(**cfg["dataset"])
    print(f"[run] built {len(train_loaders)} tasks")

    # Model.
    model = build_model(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] model={cfg['model']['name']} params={n_params:,}")

    # Baseline.
    base_kw = cfg.get("baseline_kwargs", {})
    base = build_baseline(cfg["baseline"], model, **base_kw)

    # PA-CL wrapper.
    pacl = None
    pacl_cfg = cfg.get("pacl", {}) or {}
    if pacl_cfg.get("enabled", False):
        pacl = PACL(
            model=model,
            layer_names=pacl_cfg["layer_names"],
            config=PACLConfig(
                lam=float(pacl_cfg.get("lam", 0.1)),
                beta=float(pacl_cfg.get("beta", 0.99)),
                alpha=float(pacl_cfg.get("alpha", 0.9)),
                tau=float(pacl_cfg.get("tau", 1e-4)),
                disable_projection=bool(pacl_cfg.get("disable_projection",
                                                     False)),
            ),
        )
        print(f"[run] PACL on layers {pacl_cfg['layer_names']}")
    probe_layer_names = pacl_cfg.get("layer_names") if pacl is None else None

    # Optimizer.
    o = cfg["optimizer"]
    if o["name"].lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(o["lr"]),
            momentum=float(o.get("momentum", 0.0)),
            weight_decay=float(o.get("weight_decay", 0.0)),
        )
    elif o["name"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(o["lr"]),
            weight_decay=float(o.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError(f"Unknown optimizer {o['name']}")

    t_start = time.time()
    result = train_continual(
        base=base,
        pacl=pacl,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        optimizer=optimizer,
        device=device,
        epochs_per_task=int(cfg["training"].get("epochs_per_task", 1)),
        log_every=int(cfg["training"].get("log_every", 100)),
        probe_layer_names=probe_layer_names,
        batch_size=int(cfg["dataset"].get("batch_size", 128)),
    )

    # ---------- Metrics ----------
    summary = {
        "method": method,
        "seed": seed,
        "ACC": acc_sparse(result.final_row),
        "BWT": bwt_sparse(result.diag_acc, result.final_row),
        "mean_plasticity_first": float(np.mean(result.diag_acc[:20])
                                       if len(result.diag_acc) >= 20 else
                                       np.mean(result.diag_acc)),
        "mean_plasticity_last": float(np.mean(result.diag_acc[-20:])
                                      if len(result.diag_acc) >= 20 else
                                      np.mean(result.diag_acc)),
        "plasticity_drop": (
            float(np.mean(result.diag_acc[:20]) - np.mean(result.diag_acc[-20:]))
            if len(result.diag_acc) >= 20 else 0.0
        ),
        "per_task_forgetting_mean": (
            float(np.mean(per_task_forgetting_sparse(
                result.diag_acc, result.final_row)))
        ),
        "seconds": result.seconds,
        "wall_seconds": time.time() - t_start,
        "n_tasks": len(result.diag_acc),
        "n_params": n_params,
    }
    if result.rank_traj:
        # Per-layer last/first effective rank summary.
        layers = list(result.rank_traj[0].keys())
        summary["rank_layers"] = layers
        summary["rank_first"] = {l: float(result.rank_traj[0][l]) for l in layers}
        summary["rank_last"] = {l: float(result.rank_traj[-1][l]) for l in layers}
        summary["rank_min"] = {
            l: float(min(d[l] for d in result.rank_traj)) for l in layers
        }
    print(f"[run] summary: {json.dumps(summary, indent=2)}")

    # ---------- Persist ----------
    np.save(out_dir / "diag_acc.npy", np.asarray(result.diag_acc, dtype=np.float32))
    np.save(out_dir / "final_row.npy", np.asarray(result.final_row, dtype=np.float32))
    np.save(out_dir / "train_loss.npy",
            np.asarray(result.train_loss_traj, dtype=np.float32))
    if result.rank_traj:
        with open(out_dir / "rank_traj.json", "w") as f:
            json.dump([{k: float(v) for k, v in d.items()}
                       for d in result.rank_traj], f, indent=2)
    if result.rank_floor_traj:
        with open(out_dir / "rank_floor_traj.json", "w") as f:
            json.dump([{k: float(v) for k, v in d.items()}
                       for d in result.rank_floor_traj], f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "config.yaml", "w") as f:
        # Pop our private key to keep the artifact clean.
        c = copy.deepcopy(cfg)
        c.pop("_method", None)
        yaml.safe_dump(c, f, sort_keys=False)
    print(f"[run] wrote artefacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())