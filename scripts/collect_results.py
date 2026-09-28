"""Generate the LaTeX table rows and summary CSVs from the runbook.

Reads runbook/<exp>/<method>/seed*/summary.json, computes mean +- one
sample standard deviation per method, and emits LaTeX table rows
(matching the paper's table format) plus a compact CSV.

Usage:
    python scripts/collect_results.py --benchmark cifar100_main
    python scripts/collect_results.py --benchmark tinyimagenet_main --format latex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_METHODS = [
    "erm", "ewc", "agem", "er", "derpp", "er_ace", "cls_er", "xder",
    "pacl", "pacl_er", "pacl_derpp", "pacl_er_ace", "pacl_cls_er",
    "pacl_xder",
]


def _fmt(mean: float, std: float) -> str:
    return f"${mean:.2f} \\pm {std:.2f}$"


def collect(benchmark: str, runbook: Path) -> dict:
    base = runbook / benchmark
    out = {}
    for m in _METHODS:
        accs, bwts, ranks = [], [], []
        for s in range(5):
            p = base / m / f"seed{s}" / "summary.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            accs.append(100 * d["ACC"])
            bwts.append(100 * d["BWT"])
            rl = d.get("rank_last", {})
            if isinstance(rl, dict) and rl:
                ranks.append(list(rl.values())[0])
        if not accs:
            out[m] = None
            continue
        out[m] = {
            "n": len(accs),
            "ACC": (np.mean(accs), np.std(accs, ddof=1)),
            "BWT": (np.mean(bwts), np.std(bwts, ddof=1)),
            "rank": (np.mean(ranks), np.std(ranks, ddof=1)) if ranks else None,
        }
    return out


def latex_rows(results: dict) -> str:
    rows = []
    display = {
        "erm": "ERM", "ewc": "EWC", "agem": "A-GEM", "er": "ER",
        "derpp": "DER++", "er_ace": "ER-ACE", "cls_er": "CLS-ER",
        "xder": "X-DER", "pacl": "PA-CL", "pacl_er": "PA-CL+ER",
        "pacl_derpp": "PA-CL+DER++", "pacl_er_ace": "PA-CL+ER-ACE",
        "pacl_cls_er": "PA-CL+CLS-ER", "pacl_xder": "PA-CL+X-DER",
    }
    for m in _METHODS:
        r = results.get(m)
        if r is None:
            rows.append(f"        {display[m]} & --- & --- & --- \\\\")
            continue
        rank = r.get("rank")
        rank_s = _fmt(*rank) if rank else "---"
        rows.append(
            f"        {display[m]} & {_fmt(*r['ACC'])} & {_fmt(*r['BWT'])} "
            f"& {rank_s} \\\\")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--runbook", type=Path, default=Path("runbook"))
    ap.add_argument("--format", default="latex", choices=["latex", "csv"])
    args = ap.parse_args()

    results = collect(args.benchmark, args.runbook)
    if args.format == "latex":
        print(latex_rows(results))
    else:
        import csv
        import sys
        w = csv.writer(sys.stdout)
        w.writerow(["method", "n", "ACC_mean", "ACC_std", "BWT_mean", "BWT_std", "rank_mean", "rank_std"])
        for m, r in results.items():
            if r is None:
                continue
            w.writerow([m, r["n"], f"{r['ACC'][0]:.2f}", f"{r['ACC'][1]:.2f}",
                        f"{r['BWT'][0]:.2f}", f"{r['BWT'][1]:.2f}",
                        f"{r['rank'][0]:.2f}" if r["rank"] else "",
                        f"{r['rank'][1]:.2f}" if r["rank"] else ""])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
