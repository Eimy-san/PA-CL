"""Aggregate the PoC sweep and produce the gate verdict + figures.

Inputs:
    <out_dir>/<method>/seed<k>/{summary.json, diag_acc.npy, final_row.npy,
                                rank_traj.json, [rank_floor_traj.json]}

Outputs (written to <out_dir>/_aggregate/):
    table.md            Markdown table with mean +- std for ACC, BWT,
                        plasticity drop, rank statistics.
    table.tex           Same as LaTeX (booktabs).
    plasticity.png      Plasticity-decay curves (diag_acc vs task index)
                        per method, mean +- 95% CI shaded band.
    plasticity.pdf      Vector version of the same.
    rank_traj.png/.pdf  Effective-rank trajectories per method, per layer.
    gate.json           Numeric gate evaluation against the strict rule
                        from research proposal Section 5.
    gate.txt            Human-readable PASS / PARTIAL / FAIL summary.
    raw.json            All collected scalars (per method, per seed)
                        for downstream analysis / paper tables.

Strict gate (matches research proposal Sec. 5 + user lock-in):
    PASS iff ALL of:
      (G1) erank(PA-CL, last) / erank(ERM, last) >= 1.5  on penultimate layer
      (G2) plasticity_drop(ERM) >= 5pp AND plasticity_drop(PA-CL) <= 0.5 * plasticity_drop(ERM)
      (G3) ACC(PA-CL) - ACC(ERM) >= 1.5 pp AND no overlap in mean +- std
      (G4) BWT(PA-CL) >= BWT(ERM) - 2 pp
    PARTIAL: ACC gain in [0, 1.5 pp) OR (G1) ratio in [1.2, 1.5).
    FAIL:    ACC gain < 0 OR rank ratio < 1.2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# matplotlib is optional at aggregate time on the GPU host; guard the import.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as e:  # pragma: no cover
    print(f"[agg] matplotlib unavailable ({e}); will skip figures.")
    HAS_MPL = False


# Stable color map across methods (PoC + Phase-2 baselines + PA-CL variants).
_METHOD_COLORS = {
    "erm":          "tab:gray",
    "cbp":          "tab:orange",
    "ewc":          "tab:olive",
    "er":           "tab:purple",
    "derpp":        "tab:red",
    "agem":         "tab:brown",
    "pacl":         "tab:blue",
    "pacl_erm":     "tab:blue",
    "pacl_cbp":     "tab:green",
    "pacl_er":      "tab:cyan",
    "pacl_derpp":   "tab:pink",
    "pacl_ewc":     "darkblue",
    "pacl_agem":    "deepskyblue",
}


@dataclass
class MethodResults:
    name: str
    diag_acc: np.ndarray         # (n_seeds, T)
    final_row: np.ndarray        # (n_seeds, T)
    acc: np.ndarray              # (n_seeds,)
    bwt: np.ndarray              # (n_seeds,)
    plasticity_drop: np.ndarray  # (n_seeds,)
    rank_last: Dict[str, np.ndarray]   # layer -> (n_seeds,)
    rank_first: Dict[str, np.ndarray]  # layer -> (n_seeds,)
    rank_traj: Dict[str, np.ndarray]   # layer -> (n_seeds, T)


def _load_method(method_dir: Path, name: str) -> MethodResults:
    seed_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()
                        and d.name.startswith("seed")])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed* dirs under {method_dir}")
    diag, final, acc, bwt, drop = [], [], [], [], []
    rank_first: Dict[str, List[float]] = {}
    rank_last: Dict[str, List[float]] = {}
    rank_traj_per_layer: Dict[str, List[np.ndarray]] = {}
    for sd in seed_dirs:
        with open(sd / "summary.json") as f:
            summ = json.load(f)
        diag.append(np.load(sd / "diag_acc.npy"))
        final.append(np.load(sd / "final_row.npy"))
        acc.append(float(summ["ACC"]))
        bwt.append(float(summ["BWT"]))
        drop.append(float(summ["plasticity_drop"]))
        if "rank_first" in summ:
            for L, v in summ["rank_first"].items():
                rank_first.setdefault(L, []).append(float(v))
            for L, v in summ["rank_last"].items():
                rank_last.setdefault(L, []).append(float(v))
        if (sd / "rank_traj.json").exists():
            with open(sd / "rank_traj.json") as f:
                traj = json.load(f)
            for L in traj[0].keys():
                arr = np.asarray([d[L] for d in traj], dtype=np.float32)
                rank_traj_per_layer.setdefault(L, []).append(arr)
    return MethodResults(
        name=name,
        diag_acc=np.stack(diag, axis=0),
        final_row=np.stack(final, axis=0),
        acc=np.asarray(acc, dtype=np.float64),
        bwt=np.asarray(bwt, dtype=np.float64),
        plasticity_drop=np.asarray(drop, dtype=np.float64),
        rank_first={L: np.asarray(v) for L, v in rank_first.items()},
        rank_last={L: np.asarray(v) for L, v in rank_last.items()},
        rank_traj={L: np.stack(v, axis=0) for L, v in rank_traj_per_layer.items()},
    )


def _mean_ci(x: np.ndarray, ci: float = 0.95):
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    # Approximate normal CI for small samples; we report std too.
    half = 1.96 * std / np.sqrt(max(1, x.size))
    return mean, std, mean - half, mean + half


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average of length w. Returns array of size
    max(0, x.size - w + 1) (i.e., the 'valid' convolution length)."""
    if w <= 1 or x.size < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def _smooth_xy(mu: np.ndarray, sd: np.ndarray, smooth: int):
    """Return (x, mu, sd) after centered smoothing of length `smooth`."""
    T = mu.size
    x = np.arange(T)
    if smooth <= 1 or T < smooth:
        return x, mu, sd
    mu_s = _moving_average(mu, smooth)
    sd_s = _moving_average(sd, smooth)
    # 'valid' alignment: the i-th smoothed value averages x[i:i+smooth].
    off = smooth // 2
    x_s = x[off: off + mu_s.size]
    return x_s, mu_s, sd_s


# -----------------------------------------------------------------
# Table emission
# -----------------------------------------------------------------

def _make_table(results: List[MethodResults], penult_layer: str) -> str:
    lines = []
    lines.append(
        "| Method | ACC (mean ± std) | BWT (mean ± std) | "
        "Plast.-drop (mean ± std) | erank(last, penult) (mean ± std) |"
    )
    lines.append("|---|---|---|---|---|")
    for r in results:
        acc_m, acc_s, _, _ = _mean_ci(r.acc)
        bwt_m, bwt_s, _, _ = _mean_ci(r.bwt)
        drop_m, drop_s, _, _ = _mean_ci(r.plasticity_drop)
        if penult_layer in r.rank_last:
            rl = r.rank_last[penult_layer]
            rl_m, rl_s, _, _ = _mean_ci(rl)
            rl_str = f"{rl_m:.2f} ± {rl_s:.2f}"
        else:
            rl_str = "n/a"
        lines.append(
            f"| {r.name} | "
            f"{100*acc_m:.2f} ± {100*acc_s:.2f} | "
            f"{100*bwt_m:+.2f} ± {100*bwt_s:.2f} | "
            f"{100*drop_m:.2f} ± {100*drop_s:.2f} | "
            f"{rl_str} |"
        )
    return "\n".join(lines)


def _make_latex(results: List[MethodResults], penult_layer: str) -> str:
    rows = []
    for r in results:
        acc_m, acc_s, _, _ = _mean_ci(r.acc)
        bwt_m, bwt_s, _, _ = _mean_ci(r.bwt)
        drop_m, drop_s, _, _ = _mean_ci(r.plasticity_drop)
        if penult_layer in r.rank_last:
            rl_m, rl_s, _, _ = _mean_ci(r.rank_last[penult_layer])
            rl_str = f"${rl_m:.2f} \\pm {rl_s:.2f}$"
        else:
            rl_str = "n/a"
        rows.append(
            f"  {r.name.upper()} & "
            f"${100*acc_m:.2f} \\pm {100*acc_s:.2f}$ & "
            f"${100*bwt_m:+.2f} \\pm {100*bwt_s:.2f}$ & "
            f"${100*drop_m:.2f} \\pm {100*drop_s:.2f}$ & "
            f"{rl_str} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lcccc}\n"
        "  \\toprule\n"
        "  Method & ACC (\\%) & BWT (\\%) & Plast.\\ drop (\\%) "
        "& erank (penult.) \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )


# -----------------------------------------------------------------
# Figures
# -----------------------------------------------------------------

def _plot_plasticity(results: List[MethodResults], out_path: Path,
                     smooth: int = 5) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = _METHOD_COLORS
    for r in results:
        mu = r.diag_acc.mean(axis=0)
        sd = r.diag_acc.std(axis=0, ddof=1) if r.diag_acc.shape[0] > 1 else \
            np.zeros_like(mu)
        x_s, mu_s, sd_s = _smooth_xy(mu, sd, smooth)
        c = colors.get(r.name, None)
        ax.plot(x_s, mu_s, label=r.name.upper(), color=c, linewidth=1.6)
        ax.fill_between(x_s, mu_s - sd_s, mu_s + sd_s,
                        alpha=0.2, color=c, linewidth=0)
    ax.set_xlabel("Task index $t$")
    ax.set_ylabel("New-task accuracy $a_{t,t}$")
    ax.set_title("Plasticity decay on Class-IL Split-CIFAR-100")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_rank_traj(results: List[MethodResults], out_path: Path,
                    smooth: int = 5) -> None:
    if not HAS_MPL:
        return
    # one subplot per layer present in any method
    layers: List[str] = []
    for r in results:
        for L in r.rank_traj.keys():
            if L not in layers:
                layers.append(L)
    if not layers:
        return
    fig, axes = plt.subplots(1, len(layers), figsize=(4.5 * len(layers), 3.6),
                             sharey=False)
    if len(layers) == 1:
        axes = [axes]
    colors = _METHOD_COLORS
    for ax, L in zip(axes, layers):
        for r in results:
            if L not in r.rank_traj:
                continue
            arr = r.rank_traj[L]
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mu)
            x_s, mu_s, sd_s = _smooth_xy(mu, sd, smooth)
            c = colors.get(r.name, None)
            ax.plot(x_s, mu_s, label=r.name.upper(), color=c, linewidth=1.6)
            ax.fill_between(x_s, mu_s - sd_s, mu_s + sd_s,
                            alpha=0.2, color=c, linewidth=0)
        ax.set_title(f"Effective rank: {L}")
        ax.set_xlabel("Task index $t$")
        ax.set_ylabel("$\\rho_\\ell(t)$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# -----------------------------------------------------------------
# Gate
# -----------------------------------------------------------------

def _evaluate_gate(by: Dict[str, MethodResults], penult_layer: str) -> dict:
    out: dict = {"checks": {}, "verdict": "UNKNOWN", "explanation": []}
    if "erm" not in by or "pacl" not in by:
        out["verdict"] = "INSUFFICIENT_DATA"
        out["explanation"].append("Need both ERM and PA-CL results.")
        return out

    erm = by["erm"]
    pacl = by["pacl"]

    # G1: rank ratio on penultimate layer at the end.
    g1: dict = {"name": "rank_ratio_penult_last"}
    if penult_layer in erm.rank_last and penult_layer in pacl.rank_last:
        erm_r = float(np.mean(erm.rank_last[penult_layer]))
        pacl_r = float(np.mean(pacl.rank_last[penult_layer]))
        ratio = pacl_r / max(1e-6, erm_r)
        g1.update({"erm": erm_r, "pacl": pacl_r, "ratio": ratio,
                   "threshold_pass": 1.5, "threshold_partial": 1.2,
                   "pass": ratio >= 1.5,
                   "partial": (1.2 <= ratio < 1.5)})
    else:
        g1["pass"] = False
        g1["partial"] = False
        g1["note"] = f"penultimate layer '{penult_layer}' missing in rank_last"
    out["checks"]["G1"] = g1

    # G2: plasticity drop comparison.
    g2: dict = {"name": "plasticity_drop"}
    erm_drop = float(np.mean(erm.plasticity_drop))
    pacl_drop = float(np.mean(pacl.plasticity_drop))
    g2.update({"erm_drop_pp": 100 * erm_drop, "pacl_drop_pp": 100 * pacl_drop,
               "threshold_erm_drop_pp": 5.0,
               "pass": (erm_drop >= 0.05) and (pacl_drop <= 0.5 * erm_drop)})
    out["checks"]["G2"] = g2

    # G3: ACC gain.
    g3: dict = {"name": "acc_gain"}
    acc_gap = float(np.mean(pacl.acc) - np.mean(erm.acc))
    erm_lo = float(np.mean(erm.acc) + np.std(erm.acc, ddof=1)) if erm.acc.size > 1 else float(np.mean(erm.acc))
    pacl_lo = float(np.mean(pacl.acc) - np.std(pacl.acc, ddof=1)) if pacl.acc.size > 1 else float(np.mean(pacl.acc))
    no_overlap = pacl_lo > erm_lo
    g3.update({"acc_pacl": 100 * float(np.mean(pacl.acc)),
               "acc_erm": 100 * float(np.mean(erm.acc)),
               "gap_pp": 100 * acc_gap,
               "no_overlap_mean_pm_std": bool(no_overlap),
               "threshold_pass_pp": 1.5,
               "pass": (acc_gap >= 0.015) and no_overlap,
               "partial": (0 < acc_gap < 0.015),
               "fail": acc_gap < 0})
    out["checks"]["G3"] = g3

    # G4: BWT preservation.
    g4: dict = {"name": "bwt_preservation"}
    bwt_gap = float(np.mean(pacl.bwt) - np.mean(erm.bwt))
    g4.update({"bwt_pacl_pp": 100 * float(np.mean(pacl.bwt)),
               "bwt_erm_pp": 100 * float(np.mean(erm.bwt)),
               "gap_pp": 100 * bwt_gap,
               "threshold_pass_pp": -2.0,
               "pass": bwt_gap >= -0.02})
    out["checks"]["G4"] = g4

    all_pass = all(c.get("pass", False) for c in out["checks"].values())
    any_fail = out["checks"]["G3"].get("fail", False) or \
               (g1.get("ratio", 999) < 1.2 if "ratio" in g1 else False)
    if all_pass:
        out["verdict"] = "PASS"
        out["explanation"].append(
            "All four gate conditions satisfied; lock PA-CL and proceed "
            "to the full main-table sweep.")
    elif any_fail:
        out["verdict"] = "FAIL"
        out["explanation"].append(
            "Either ACC gap is negative or rank ratio < 1.2; pivot per "
            "research-proposal contingency table (F1/F2/F6).")
    else:
        out["verdict"] = "PARTIAL"
        out["explanation"].append(
            "Some checks satisfied but not all; either tune lambda / "
            "probe-layer selection and re-run, or accept a weakened "
            "claim ('PA-CL improves rank/plasticity without ACC gain')."
        )
    return out


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--methods", nargs="+",
                    default=["erm", "cbp", "pacl"])
    ap.add_argument("--penult_layer", default="hidden.h2",
                    help="Layer used as 'penultimate' for the gate.")
    args = ap.parse_args()

    agg_dir = args.out_dir / "_aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    results: List[MethodResults] = []
    by_name: Dict[str, MethodResults] = {}
    for m in args.methods:
        mdir = args.out_dir / m
        if not mdir.exists():
            print(f"[agg] WARN: missing method dir {mdir}; skipping.")
            continue
        r = _load_method(mdir, m)
        results.append(r)
        by_name[m] = r
        print(f"[agg] {m}: {r.diag_acc.shape[0]} seeds x "
              f"{r.diag_acc.shape[1]} tasks")

    if not results:
        print("[agg] No results found.", file=sys.stderr)
        return 1

    # Tables.
    md = _make_table(results, args.penult_layer)
    tex = _make_latex(results, args.penult_layer)
    (agg_dir / "table.md").write_text(md + "\n")
    (agg_dir / "table.tex").write_text(tex)

    # Figures.
    _plot_plasticity(results, agg_dir / "plasticity")
    _plot_rank_traj(results, agg_dir / "rank_traj")

    # Gate.
    gate = _evaluate_gate(by_name, args.penult_layer)
    (agg_dir / "gate.json").write_text(json.dumps(gate, indent=2))

    # Human-readable summary.
    lines = ["PA-CL PoC Aggregate Report", "=" * 36, "", "Table:", "", md, ""]
    lines.append(f"Gate verdict: {gate['verdict']}")
    for k, v in gate["checks"].items():
        lines.append(f"  [{k}] {v}")
    for exp in gate["explanation"]:
        lines.append(f"  -> {exp}")
    (agg_dir / "gate.txt").write_text("\n".join(lines) + "\n")

    # Raw dump.
    raw = {
        m: {
            "acc": r.acc.tolist(),
            "bwt": r.bwt.tolist(),
            "plasticity_drop": r.plasticity_drop.tolist(),
            "rank_first": {L: v.tolist() for L, v in r.rank_first.items()},
            "rank_last": {L: v.tolist() for L, v in r.rank_last.items()},
        }
        for m, r in by_name.items()
    }
    (agg_dir / "raw.json").write_text(json.dumps(raw, indent=2))

    print("\n" + "\n".join(lines))
    print(f"\n[agg] artefacts in: {agg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())