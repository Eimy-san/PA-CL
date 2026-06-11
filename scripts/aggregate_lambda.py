"""Aggregate the lambda-sweep results (PA-CL+ER, Split-CIFAR-100).

Inputs: <out_dir>/lam_<value>/seed*/{summary.json, ...}

Outputs (written to <out_dir>/_aggregate/):
    table.md            Markdown table: lambda | ACC | BWT | rank
    table.tex           LaTeX (booktabs) version of the same
    sweep.png           Two-panel plot: ACC vs lambda, rank vs lambda
    sweep.pdf           Vector version
    raw.json            All scalars per lambda
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as e:  # pragma: no cover
    print(f"[lam-agg] matplotlib unavailable ({e}); will skip figures.")
    HAS_MPL = False


def _parse_lambda_tag(name: str) -> float:
    m = re.match(r"lam_([\d.]+)", name)
    if not m:
        raise ValueError(f"unexpected tag name: {name}")
    return float(m.group(1))


def _load_cells(out_dir: Path) -> List[Tuple[float, dict]]:
    cells: List[Tuple[float, dict]] = []
    for tag_dir in sorted(out_dir.iterdir()):
        if not tag_dir.is_dir():
            continue
        if not tag_dir.name.startswith("lam_"):
            continue
        try:
            lam = _parse_lambda_tag(tag_dir.name)
        except ValueError:
            continue
        seed_dirs = sorted([d for d in tag_dir.iterdir() if d.is_dir()
                            and d.name.startswith("seed")])
        if not seed_dirs:
            continue
        with open(seed_dirs[0] / "summary.json") as f:
            summ = json.load(f)
        cells.append((lam, summ))
    cells.sort(key=lambda x: x[0])
    return cells


def _make_table_md(cells: List[Tuple[float, dict]]) -> str:
    lines = []
    lines.append("| $\\lambda$ | ACC (\\%) | BWT (\\%) | erank (penult.) |")
    lines.append("|---|---|---|---|")
    for lam, s in cells:
        acc = 100 * float(s["ACC"])
        bwt = 100 * float(s["BWT"])
        rl = next(iter(s.get("rank_last", {}).values()), float("nan"))
        lines.append(f"| {lam:g} | {acc:.2f} | {bwt:+.2f} | {rl:.2f} |")
    return "\n".join(lines)


def _make_table_tex(cells: List[Tuple[float, dict]]) -> str:
    rows = []
    for lam, s in cells:
        acc = 100 * float(s["ACC"])
        bwt = 100 * float(s["BWT"])
        rl = next(iter(s.get("rank_last", {}).values()), float("nan"))
        rows.append(f"  ${lam:g}$ & ${acc:.2f}$ & ${bwt:+.2f}$ & ${rl:.2f}$ \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{cccc}\n"
        "  \\toprule\n"
        "  $\\lambda$ & ACC (\\%) & BWT (\\%) & erank (penult.) \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )


def _plot_sweep(cells: List[Tuple[float, dict]], out_path: Path) -> None:
    if not HAS_MPL or not cells:
        return
    lams = np.array([c[0] for c in cells], dtype=float)
    accs = np.array([100 * float(c[1]["ACC"]) for c in cells], dtype=float)
    bwts = np.array([100 * float(c[1]["BWT"]) for c in cells], dtype=float)
    ranks = np.array([
        next(iter(c[1].get("rank_last", {}).values()), float("nan"))
        for c in cells
    ], dtype=float)
    # X-axis is log-ish, but 0 is a problem. Map 0 -> position 0 on a
    # symlog-style axis; plot the rest on log.
    xticks = lams.copy()
    xlabels = [f"{l:g}" for l in lams]
    x = np.arange(len(lams))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax1.plot(x, accs, "o-", color="tab:blue", linewidth=1.8, markersize=6,
             label="ACC (\\%)")
    ax1.set_xlabel("$\\lambda$")
    ax1.set_ylabel("Average accuracy (\\%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("ACC vs $\\lambda$")

    ax2.plot(x, ranks, "s-", color="tab:red", linewidth=1.8, markersize=6,
             label="erank (penult.)")
    ax2.set_xlabel("$\\lambda$")
    ax2.set_ylabel("Effective rank (penultimate)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Effective rank vs $\\lambda$")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    agg_dir = args.out_dir / "_aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    cells = _load_cells(args.out_dir)
    if not cells:
        print("[lam-agg] no lam_* directories found", file=sys.stderr)
        return 1
    print(f"[lam-agg] loaded {len(cells)} cells: lambdas={[c[0] for c in cells]}")

    md = _make_table_md(cells)
    tex = _make_table_tex(cells)
    (agg_dir / "table.md").write_text(md + "\n")
    (agg_dir / "table.tex").write_text(tex)

    _plot_sweep(cells, agg_dir / "sweep")

    raw = {
        f"lam_{lam:g}": {
            "ACC": float(s["ACC"]),
            "BWT": float(s["BWT"]),
            "rank_last": dict(s.get("rank_last", {})),
            "seconds": float(s.get("seconds", 0)),
        }
        for lam, s in cells
    }
    (agg_dir / "raw.json").write_text(json.dumps(raw, indent=2))

    print("\n" + md + "\n")
    print(f"[lam-agg] artefacts written to {agg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())