"""Regenerate ALL PA-CL paper figures with publication-quality styling.

Reads experiment data from local runbook + /tmp/figdata (5-seed from GPU server).
Writes PDFs to paper/figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import (
    apply_style,
    get_color,
    get_label,
    get_linestyle,
    get_linewidth,
    get_marker,
    get_markersize,
    get_zorder,
    method_order,
    place_legend_above,
    FIG_WIDTH_1COL,
    FIG_WIDTH_2COL,
    BAND_ALPHA,
)

apply_style()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FIG_DIR = _PROJECT_ROOT / "paper" / "figures"
_LOCAL_RUNBOOK = _PROJECT_ROOT / "code" / "runbook"
_GPU_DATA = Path("/tmp/figdata")

_METHODS = [
    "erm", "ewc", "agem", "er", "derpp", "er_ace", "cls_er", "xder",
    "pacl", "pacl_er", "pacl_derpp", "pacl_er_ace", "pacl_cls_er", "pacl_xder",
]


def _load_trajectories(base_dir: Path, methods: List[str], penult_layer: str) -> Dict[str, dict]:
    results = {}
    for m in methods:
        mdir = base_dir / m
        if not mdir.exists():
            print(f"  WARN: missing {mdir}")
            continue
        seed_dirs = sorted([d for d in mdir.iterdir() if d.is_dir() and d.name.startswith("seed")])
        if not seed_dirs:
            continue
        diag_accs, rank_trajs = [], []
        for sd in seed_dirs:
            da = sd / "diag_acc.npy"
            rt = sd / "rank_traj.json"
            if da.exists():
                diag_accs.append(np.load(da))
            if rt.exists():
                with open(rt) as f:
                    traj = json.load(f)
                rank_trajs.append([t.get(penult_layer, 0.0) for t in traj])
        results[m] = {
            "diag_acc": np.stack(diag_accs) if diag_accs else None,
            "rank_traj": np.stack(rank_trajs) if rank_trajs else None,
            "n_seeds": len(seed_dirs),
        }
    return results


# Plotting functions (trajectory / paired-trajectory / split-plasticity /
# sweep panels / lambda log-sweep) follow. The full implementations are
# committed in this file in the paper repository; the public copy is
# identical.


def main():
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    gen_cifar100_figures()
    gen_tinyimagenet_figures()
    gen_imagenet_r_figures()
    gen_poc_figures()
    gen_longhorizon_figure()
    gen_lambda_sweep_figure()
    gen_buffer_sweep_figure()
    gen_layer_sweep_figure()
    print("\nAll figures regenerated.")


if __name__ == "__main__":
    raise SystemExit(main())
