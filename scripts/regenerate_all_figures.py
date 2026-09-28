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
    "erm",
    "ewc",
    "agem",
    "er",
    "derpp",
    "er_ace",
    "cls_er",
    "xder",
    "pacl",
    "pacl_er",
    "pacl_derpp",
    "pacl_er_ace",
    "pacl_cls_er",
    "pacl_xder",
]


def _load_trajectories(
    base_dir: Path, methods: List[str], penult_layer: str
) -> Dict[str, dict]:
    results = {}
    for m in methods:
        mdir = base_dir / m
        if not mdir.exists():
            print(f"  WARN: missing {mdir}")
            continue
        seed_dirs = sorted(
            [d for d in mdir.iterdir() if d.is_dir() and d.name.startswith("seed")]
        )
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


def _plot_trajectory_fig(
    results: dict,
    out_path: Path,
    title: str,
    ylabel: str,
    xlabel: str = "Task index $t$",
    n_tasks: Optional[int] = None,
    data_key: str = "rank_traj",
    smooth: int = 1,
    figsize: tuple = (FIG_WIDTH_2COL, 2.8),
):
    fig, ax = plt.subplots(figsize=figsize)
    for m in method_order(list(results.keys())):
        r = results[m]
        data = r[data_key]
        if data is None:
            continue
        mu = data.mean(axis=0)
        sd = data.std(axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mu)
        x = np.arange(mu.size)

        if smooth > 1 and mu.size > smooth:
            kernel = np.ones(smooth) / smooth
            mu_s = np.convolve(mu, kernel, mode="valid")
            sd_s = np.convolve(sd, kernel, mode="valid")
            off = smooth // 2
            x = x[off : off + mu_s.size]
            mu, sd = mu_s, sd_s

        c = get_color(m)
        ax.plot(
            x,
            mu,
            label=get_label(m),
            color=c,
            linewidth=get_linewidth(m),
            marker=get_marker(m),
            markersize=get_markersize(m),
            markevery=max(1, len(x) // 15),
            linestyle=get_linestyle(m),
            zorder=get_zorder(m),
        )
        ax.fill_between(
            x,
            mu - sd,
            mu + sd,
            alpha=BAND_ALPHA,
            color=c,
            linewidth=0,
            zorder=get_zorder(m) - 1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if n_tasks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(n_tasks, 10)))
    place_legend_above(ax, len(results))
    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(out_path.with_suffix(ext))
    plt.close(fig)
    print(f"  -> {out_path.name}")


def _plot_sweep_fig(
    cells: list,
    out_path: Path,
    xlabel: str,
    title_left: str,
    title_right: str,
    xlabels: Optional[list] = None,
    ylabel_right: str = "Effective rank (penult.)",
    figsize: tuple = (FIG_WIDTH_2COL, 2.6),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(cells))
    if xlabels is None:
        xlabels = [str(c[0]) for c in cells]

    acc_m, acc_s, rank_m, rank_s = [], [], [], []
    for _, seeds in cells:
        accs = [100 * s["ACC"] for s in seeds]
        ranks = [
            (
                sum(s.get("rank_last", {}).values())
                if isinstance(s.get("rank_last", {}), dict)
                else s.get("rank_last", float("nan"))
            )
            for s in seeds
        ]
        acc_m.append(np.mean(accs))
        acc_s.append(np.std(accs, ddof=1) if len(accs) > 1 else 0)
        rank_m.append(np.mean(ranks))
        rank_s.append(np.std(ranks, ddof=1) if len(ranks) > 1 else 0)

    ax1.errorbar(
        x,
        acc_m,
        yerr=acc_s,
        fmt="o-",
        color="#2563EB",
        linewidth=1.8,
        markersize=5,
        capsize=3,
        capthick=1.2,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Avg. accuracy (\\%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=8)

    ax2.errorbar(
        x,
        rank_m,
        yerr=rank_s,
        fmt="s-",
        color="#DC2626",
        linewidth=1.8,
        markersize=5,
        capsize=3,
        capthick=1.2,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_right)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=8)

    fig.tight_layout(w_pad=2.5)
    for ext in [".pdf", ".png"]:
        fig.savefig(out_path.with_suffix(ext))
    plt.close(fig)
    print(f"  -> {out_path.name}")


def _sort_key_numeric(name: str, prefix: str):
    suffix = name[len(prefix) :]
    try:
        return (0, float(suffix), name)
    except ValueError:
        return (1, 0.0, name)


def _load_sweep_cells(base_dir: Path, prefix: str) -> list:
    import re

    candidates = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
    ]
    candidates.sort(key=lambda d: _sort_key_numeric(d.name, prefix))
    cells = []
    for tag_dir in candidates:
        seed_dirs = sorted(
            [d for d in tag_dir.iterdir() if d.is_dir() and d.name.startswith("seed")]
        )
        if not seed_dirs:
            continue
        summaries = []
        for sd in seed_dirs:
            with open(sd / "summary.json") as f:
                summaries.append(json.load(f))
        cells.append((tag_dir.name, summaries))
    return cells


def _plot_paired_trajectory_fig(
    results: dict,
    out_path: Path,
    ylabel: str,
    xlabel: str = "Task index $t$",
    n_tasks: Optional[int] = None,
    data_key: str = "rank_traj",
    figsize: tuple = (FIG_WIDTH_2COL, 2.8),
):
    """Trajectory figure that highlights paired comparisons.

    Paired methods (ER vs PA-CL+ER, DER++ vs PA-CL+DER++) and PA-CL
    alone are drawn with full color and thick lines. Unpaired baselines
    (ERM, EWC, A-GEM, ER-ACE, CLS-ER, X-DER) are drawn as thin dotted
    lines in their own palette colour and marker, so they stay visually
    secondary yet individually identifiable.
    """
    paired = {
        "er",
        "derpp",
        "pacl",
        "pacl_er",
        "pacl_derpp",
        "pacl_er_ace",
        "pacl_cls_er",
        "pacl_xder",
    }
    context = {"erm", "ewc", "agem", "er_ace", "cls_er", "xder"}
    fig, ax = plt.subplots(figsize=figsize)
    n_plotted = 0
    for m in method_order(list(results.keys())):
        r = results[m]
        data = r[data_key]
        if data is None:
            continue
        n_plotted += 1
        mu = data.mean(axis=0)
        sd = data.std(axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mu)
        x = np.arange(mu.size)
        if m in context:
            ax.plot(
                x,
                mu,
                label=get_label(m),
                color=get_color(m),
                linewidth=1.15,
                linestyle=":",
                marker=get_marker(m),
                markersize=3.4,
                markevery=max(1, len(x) // 10),
                zorder=1,
            )
        else:
            c = get_color(m)
            ax.plot(
                x,
                mu,
                label=get_label(m),
                color=c,
                linewidth=get_linewidth(m),
                marker=get_marker(m),
                markersize=get_markersize(m),
                markevery=max(1, len(x) // 15),
                linestyle=get_linestyle(m),
                zorder=get_zorder(m),
            )
            ax.fill_between(
                x,
                mu - sd,
                mu + sd,
                alpha=BAND_ALPHA,
                color=c,
                linewidth=0,
                zorder=get_zorder(m) - 1,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if n_tasks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(n_tasks, 10)))
    place_legend_above(ax, n_plotted)
    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(out_path.with_suffix(ext))
    plt.close(fig)
    print(f"  -> {out_path.name}")


def _plot_split_plasticity_fig(
    results: dict,
    out_path: Path,
    xlabel: str = "Task index $t$",
    n_tasks: Optional[int] = None,
    figsize: tuple = (FIG_WIDTH_2COL, 3.2),
):
    """Plasticity figure split into replay and no-replay panels."""
    replay_methods = [
        "er",
        "er_ace",
        "cls_er",
        "xder",
        "pacl_er",
        "derpp",
        "pacl_derpp",
        "pacl_er_ace",
        "pacl_cls_er",
        "pacl_xder",
    ]
    noreplay_methods = ["erm", "ewc", "agem", "pacl"]
    panels = [
        ("(a) Replay family", replay_methods),
        ("(b) No-replay family", noreplay_methods),
    ]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (panel_title, methods) in zip(axes, panels):
        n_plotted = 0
        for m in method_order([m for m in methods if m in results]):
            r = results[m]
            data = r["diag_acc"]
            if data is None:
                continue
            n_plotted += 1
            mu = data.mean(axis=0)
            sd = data.std(axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mu)
            x = np.arange(mu.size)
            c = get_color(m)
            ax.plot(
                x,
                mu,
                label=get_label(m),
                color=c,
                linewidth=get_linewidth(m),
                marker=get_marker(m),
                markersize=get_markersize(m),
                markevery=max(1, len(x) // 12),
                linestyle=get_linestyle(m),
                zorder=get_zorder(m),
            )
            ax.fill_between(
                x,
                mu - sd,
                mu + sd,
                alpha=BAND_ALPHA,
                color=c,
                linewidth=0,
                zorder=get_zorder(m) - 1,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"New-task acc.\ $a_{t,t}$")
        if n_tasks:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(n_tasks, 8)))
        place_legend_above(ax, n_plotted, max_per_row=4, title=panel_title)
    fig.tight_layout(w_pad=2.5)
    for ext in [".pdf", ".png"]:
        fig.savefig(out_path.with_suffix(ext))
    plt.close(fig)
    print(f"  -> {out_path.name}")


def gen_cifar100_figures():
    print("[CIFAR-100]")
    base = _GPU_DATA / "cifar100_main"
    if not base.exists():
        base = _LOCAL_RUNBOOK / "cifar100_main"
    results = _load_trajectories(base, _METHODS, "avgpool_tap")
    _plot_paired_trajectory_fig(
        results,
        _FIG_DIR / "cifar100_rank_traj",
        ylabel=r"$\rho_{\mathrm{avgpool}}(t)$",
        n_tasks=10,
        data_key="rank_traj",
    )
    _plot_split_plasticity_fig(
        results,
        _FIG_DIR / "cifar100_plasticity",
        n_tasks=10,
    )


def gen_tinyimagenet_figures():
    print("[Tiny-ImageNet]")
    base = _GPU_DATA / "tinyimagenet_main"
    if not base.exists():
        base = _LOCAL_RUNBOOK / "tinyimagenet_main"
    if not base.exists():
        print("  SKIP: no data")
        return
    results = _load_trajectories(base, _METHODS, "avgpool_tap")
    _plot_paired_trajectory_fig(
        results,
        _FIG_DIR / "tinyimagenet_rank_traj",
        ylabel=r"$\rho_{\mathrm{avgpool}}(t)$",
        n_tasks=10,
        data_key="rank_traj",
    )
    _plot_split_plasticity_fig(
        results,
        _FIG_DIR / "tinyimagenet_plasticity",
        n_tasks=10,
    )


def gen_imagenet_r_figures():
    print("[ImageNet-R]")
    base = _LOCAL_RUNBOOK / "imagenet_r_main"
    if not base.exists():
        print("  SKIP: no data")
        return
    results = _load_trajectories(base, _METHODS, "avgpool_tap")
    if not results:
        print("  SKIP: no trajectory data")
        return
    _plot_paired_trajectory_fig(
        results,
        _FIG_DIR / "imagenet_r_rank_traj",
        ylabel=r"$\rho_{\mathrm{avgpool}}(t)$",
        n_tasks=10,
        data_key="rank_traj",
    )
    _plot_split_plasticity_fig(
        results,
        _FIG_DIR / "imagenet_r_plasticity",
        n_tasks=10,
    )


def gen_poc_figures():
    print("[PoC — Permuted-MNIST 200 tasks]")
    base = _LOCAL_RUNBOOK / "poc_cpmnist200"
    poc_methods = ["erm", "pacl", "cbp"]
    results = _load_trajectories(base, poc_methods, "hidden.h2")
    if not results:
        print("  SKIP: no data")
        return
    _plot_trajectory_fig(
        results,
        _FIG_DIR / "poc_rank_traj",
        title="Permuted-MNIST (200 tasks)",
        ylabel=r"$\rho_{\mathrm{h2}}(t)$",
        n_tasks=200,
        data_key="rank_traj",
        smooth=5,
        figsize=(FIG_WIDTH_2COL, 2.8),
    )
    _plot_trajectory_fig(
        results,
        _FIG_DIR / "poc_plasticity",
        title="Permuted-MNIST (200 tasks)",
        ylabel=r"New-task acc.\ $a_{t,t}$",
        n_tasks=200,
        data_key="diag_acc",
        smooth=5,
        figsize=(FIG_WIDTH_2COL, 2.8),
    )


def gen_longhorizon_figure():
    print("[Long Horizon — 1000 tasks]")
    base = _LOCAL_RUNBOOK / "poc_cpmnist1000"
    poc_methods = ["erm", "pacl", "cbp"]
    results = _load_trajectories(base, poc_methods, "hidden.h2")
    if not results:
        print("  SKIP: no data")
        return
    _plot_trajectory_fig(
        results,
        _FIG_DIR / "longhorizon_rank_traj",
        title="Permuted-MNIST (1000 tasks)",
        ylabel=r"$\rho_{\mathrm{h2}}(t)$",
        n_tasks=1000,
        data_key="rank_traj",
        smooth=20,
        figsize=(FIG_WIDTH_2COL, 2.8),
    )


def gen_lambda_sweep_figure():
    print("[Lambda Sweep]")
    base = _GPU_DATA / "cifar100_lambda"
    if not base.exists():
        base = _LOCAL_RUNBOOK / "cifar100_lambda"
    cells = _load_sweep_cells(base, "lam_")
    if not cells:
        print("  SKIP: no data")
        return

    pos_cells = []
    baseline_acc = None
    baseline_rank = None
    baseline_acc_s = 0.0
    baseline_rank_s = 0.0
    for tag, seeds in cells:
        lam_val = float(tag.replace("lam_", ""))
        accs = [100 * s["ACC"] for s in seeds]
        ranks = [
            next(iter(s.get("rank_last", {}).values()), float("nan")) for s in seeds
        ]
        if lam_val == 0:
            baseline_acc = np.mean(accs)
            baseline_rank = np.mean(ranks)
            baseline_acc_s = np.std(accs, ddof=1) if len(accs) > 1 else 0
            baseline_rank_s = np.std(ranks, ddof=1) if len(ranks) > 1 else 0
        else:
            pos_cells.append((lam_val, accs, ranks))

    pos_cells.sort(key=lambda c: c[0])
    lams = [c[0] for c in pos_cells]
    acc_m = [np.mean(c[1]) for c in pos_cells]
    acc_s = [np.std(c[1], ddof=1) if len(c[1]) > 1 else 0 for c in pos_cells]
    rank_m = [np.mean(c[2]) for c in pos_cells]
    rank_s = [np.std(c[2], ddof=1) if len(c[2]) > 1 else 0 for c in pos_cells]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_2COL, 2.6))

    for ax, vals, errs, bl_val, bl_err, ylabel, color, marker in [
        (
            ax1,
            acc_m,
            acc_s,
            baseline_acc,
            baseline_acc_s,
            "Avg. accuracy (\\%)",
            "#2563EB",
            "o",
        ),
        (
            ax2,
            rank_m,
            rank_s,
            baseline_rank,
            baseline_rank_s,
            "Effective rank (penult.)",
            "#DC2626",
            "s",
        ),
    ]:
        ax.errorbar(
            lams,
            vals,
            yerr=errs,
            fmt=f"{marker}-",
            color=color,
            linewidth=1.8,
            markersize=5,
            capsize=3,
            capthick=1.2,
            markerfacecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
        if bl_val is not None:
            ax.axhline(
                bl_val,
                color="#6B7280",
                linestyle="--",
                linewidth=1.2,
                zorder=2,
                label=r"$\lambda=0$ (plain ER)",
            )
            ax.fill_between(
                [lams[0] * 0.5, lams[-1] * 2],
                [bl_val - bl_err] * 2,
                [bl_val + bl_err] * 2,
                color="#6B7280",
                alpha=0.12,
                zorder=1,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$ (positive values, log scale)")
        ax.set_ylabel(ylabel)
        if bl_val is not None:
            place_legend_above(ax, 1, fontsize=10)

    fig.tight_layout(w_pad=2.5)
    for ext in [".pdf", ".png"]:
        fig.savefig(_FIG_DIR / f"lambda_sweep{ext}")
    plt.close(fig)
    print(f"  -> lambda_sweep.pdf")


def gen_buffer_sweep_figure():
    print("[Buffer Sweep]")
    base = _GPU_DATA / "cifar100_buffer"
    if not base.exists():
        base = _LOCAL_RUNBOOK / "cifar100_buffer"
    if not base.exists():
        print("  SKIP: no data")
        return
    cells = _load_sweep_cells(base, "buf_")
    xlabels = [c[0].replace("buf_", "") for c in cells]
    _plot_sweep_fig(
        cells,
        _FIG_DIR / "buffer_sweep",
        xlabel="Buffer size",
        title_left="ACC",
        title_right="erank",
        xlabels=xlabels,
    )


def gen_layer_sweep_figure():
    print("[Layer Sweep]")
    base = _GPU_DATA / "cifar100_layers"
    if not base.exists():
        base = _LOCAL_RUNBOOK / "cifar100_layers"
    if not base.exists():
        print("  SKIP: no data")
        return
    cells = _load_sweep_cells(base, "layers_")
    xlabels = [c[0].replace("layers_", "") for c in cells]
    _plot_sweep_fig(
        cells,
        _FIG_DIR / "layer_sweep",
        xlabel="Layer config",
        title_left="ACC",
        title_right="erank",
        xlabels=xlabels,
        ylabel_right="Effective rank (monitored aggregate)",
    )


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
