"""Theorem-1, Theorem-3 and Remark-4 diagnostic experiment.

Two-layer ReLU network: f(x) = U * phi(Wx).
Verifies three theoretical results:

(a) Theorem 1 (capacity-collapse): when rank(H) < rank(Y*), the
    attainable squared loss is bounded below by the Eckart-Young
    residual of Y* at rank(H).
(b) Theorem 3 (approximate collapse): for H with full numerical rank
    but a small spectral tail, a capacity-limited head (||U||_op <= B)
    cannot fit the target, and the loss is bounded below by a function
    of the spectral tail R_k(H) and B.
(c) Remark 4 (erank as spectral-concentration proxy): the effective
    rank erank(H) empirically tracks the index k at which the spectral
    tail R_k(H) transitions from large to small, validating erank as a
    practical proxy for spectral concentration.

Runs entirely on CPU. No GPU required.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style

apply_style()
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 42
DIMS = dict(n=30, d=25, K=15, r=12, m=500)
FIXED_B = 0.5
FIG_OUT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "paper",
    "figures",
    "theorem_diagnostic.pdf",
)


def effective_rank(H: np.ndarray) -> float:
    s = np.linalg.svd(H, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def spectral_tail(H: np.ndarray, k: int) -> float:
    s = np.linalg.svd(H, compute_uv=False)
    if k >= len(s):
        return 0.0
    return float(np.sqrt(np.sum(s[k:] ** 2)))


def eckart_young_residual(Y: np.ndarray, k: int) -> float:
    s = np.linalg.svd(Y, compute_uv=False)
    if k >= len(s):
        return 0.0
    return float(np.sqrt(np.sum(s[k:] ** 2)))


def project_spectral_norm(V: np.ndarray, B: float) -> np.ndarray:
    """Project matrix V onto {||V||_op <= B} via SVD clipping."""
    W, s, Zt = np.linalg.svd(V, full_matrices=False)
    s = np.minimum(s, B)
    return W @ np.diag(s) @ Zt


def constrained_loss(
    H: np.ndarray, Y: np.ndarray, B: float, n_iters: int = 2000
) -> float:
    """Solve min_{||U||_op <= B} ||UH - Y||_F^2 / m via projected gradient.

    Uses the SVD decomposition H = P Sigma Q^T to transform the problem
    into min_{||V||_op <= B} ||V Sigma - Y'||_F^2 + const where V = U P,
    Y' = Y Q.  The constant is ||Y*(I - Q Q^T)||_F^2, the component of
    Y* orthogonal to the row space of H.
    """
    m = Y.shape[1]
    P, sigma, Qt = np.linalg.svd(H, full_matrices=False)
    d = len(sigma)
    sigma = np.maximum(sigma, 1e-12)

    Yp = Y @ Qt.T  # Y' = Y Q, shape (K, d)
    const_part = float(np.sum(Y**2) - np.sum(Yp**2)) / m

    V_star = Yp / sigma[np.newaxis, :]  # unconstrained optimum

    if np.linalg.norm(V_star, ord=2) <= B:
        return const_part

    lr = 0.5 / (sigma.max() ** 2 + 1e-12)
    V = project_spectral_norm(V_star, B)
    for _ in range(n_iters):
        grad = 2 * (V - V_star) @ np.diag(sigma**2)
        V = project_spectral_norm(V - lr * grad, B)

    residual = V @ np.diag(sigma) - Yp
    loss = float(np.sum(residual**2) / m)
    return loss + const_part


def unconstrained_loss(H: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Unconstrained optimal U* = Y H^+, return (loss/m, ||U*||_op)."""
    m = Y.shape[1]
    U_star, _, _, _ = np.linalg.lstsq(H.T, Y.T, rcond=None)
    U_star = U_star.T
    loss = float(np.sum((U_star @ H - Y) ** 2) / m)
    B = float(np.linalg.norm(U_star, ord=2))
    return loss, B


def run_part_a(rng: np.random.Generator) -> dict:
    """Theorem 1: vary exact rank of H, measure loss vs Eckart-Young bound."""
    n, d, K, r, m = DIMS["n"], DIMS["d"], DIMS["K"], DIMS["r"], DIMS["m"]

    X = rng.standard_normal((n, m)) / np.sqrt(n)
    A = rng.standard_normal((K, r)) / np.sqrt(r)
    B_mat = rng.standard_normal((r, m)) / np.sqrt(m)
    Y_star = A @ B_mat
    Y_star_svals = np.linalg.svd(Y_star, compute_uv=False)

    W = rng.standard_normal((d, n)) / np.sqrt(n)
    H_full = np.maximum(W @ X, 0.0)
    U_f, S_f, Vh_f = np.linalg.svd(H_full, full_matrices=False)

    results = {"ranks": [], "measured_loss": [], "ey_bound": [], "erank": []}
    for k in range(1, d + 1):
        H_k = U_f[:, :k] @ np.diag(S_f[:k]) @ Vh_f[:k, :]
        loss, _ = unconstrained_loss(H_k, Y_star)
        ey = float(np.sum(Y_star_svals[k:] ** 2) / m) if k < len(Y_star_svals) else 0.0
        er = effective_rank(H_k)
        results["ranks"].append(k)
        results["measured_loss"].append(loss)
        results["ey_bound"].append(ey)
        results["erank"].append(er)

    return results


def run_part_b(rng: np.random.Generator) -> dict:
    """Theorem 3: full numerical rank, varying spectral tail, fixed B.

    Constructs H = U_b diag(sigma) Vh_b from a real ReLU(WX), then
    rescales singular values to control the spectral tail while keeping
    full numerical rank.  Uses a FIXED capacity-limited head (||U||_op
    <= B) so the constrained loss depends on the singular values.

    Two sweeps are combined:
      * k0 sweep (eps=1e-4): vary the number of large singular values
        to get a range of erank and k* values for panel (c).
      * eps sweep (k0=10): vary the tail magnitude at a fixed k0 to
        show the bound becoming trivial as the tail grows.
    """
    n, d, K, r, m = DIMS["n"], DIMS["d"], DIMS["K"], DIMS["r"], DIMS["m"]
    B = FIXED_B

    X = rng.standard_normal((n, m)) / np.sqrt(n)
    A = rng.standard_normal((K, r)) / np.sqrt(r)
    B_mat = rng.standard_normal((r, m)) / np.sqrt(m)
    Y_star = A @ B_mat
    Y_star_svals = np.linalg.svd(Y_star, compute_uv=False)

    W = rng.standard_normal((d, n)) / np.sqrt(n)
    H_base = np.maximum(W @ X, 0.0)
    U_b, S_b, Vh_b = np.linalg.svd(H_base, full_matrices=False)

    configs = []
    for k0 in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24]:
        configs.append((k0, 1e-4))
    for eps in [1e-6, 1e-3, 1e-2, 1e-1, 3e-1, 1.0]:
        if (10, eps) not in configs:
            configs.append((10, eps))

    results = {
        "k0": [],
        "tail_eps": [],
        "constrained_loss": [],
        "approx_bound": [],
        "best_k": [],
        "erank": [],
        "R_k_at_erank": [],
        "B_fixed": B,
    }

    for k0, eps in configs:
        sigma_new = S_b.copy()
        sigma_new[k0:] = eps * S_b[k0:]
        H = U_b @ np.diag(sigma_new) @ Vh_b

        c_loss = constrained_loss(H, Y_star, B)
        er = effective_rank(H)
        k_er = max(1, int(np.floor(er)))
        R_er = spectral_tail(H, k_er)

        best_bound = 0.0
        best_k = 0
        for k in range(1, min(r, d)):
            Rk = spectral_tail(H, k)
            ey_res = eckart_young_residual(Y_star, k)
            bound = max(0.0, ey_res - B * Rk) ** 2 / m
            if bound > best_bound:
                best_bound = bound
                best_k = k

        results["k0"].append(k0)
        results["tail_eps"].append(eps)
        results["constrained_loss"].append(c_loss)
        results["approx_bound"].append(best_bound)
        results["best_k"].append(best_k)
        results["erank"].append(er)
        results["R_k_at_erank"].append(R_er)

    return results


def make_figure(part_a: dict, part_b: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.0))

    # (a) Theorem 1
    ax = axes[0]
    ranks = np.array(part_a["ranks"])
    ax.plot(
        ranks,
        part_a["measured_loss"],
        "o-",
        color="#059669",
        markersize=3.5,
        linewidth=1.8,
        label="Measured loss",
        zorder=3,
    )
    ax.plot(
        ranks,
        part_a["ey_bound"],
        "s--",
        color="#DC2626",
        markersize=3.5,
        linewidth=1.4,
        label="Eckart-Young bound",
        zorder=2,
    )
    r = DIMS["r"]
    ax.axvline(r, color="#6B7280", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(r + 1.6, ax.get_ylim()[1] * 0.78, f"$r={r}$", fontsize=9, color="#6B7280")
    ax.set_xlabel(r"Induced rank $k$ of $H$")
    ax.set_ylabel(r"Squared loss $/\, m$")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        title="(a) Thm. 1: exact-rank collapse",
        title_fontsize=8.5,
        frameon=False,
        borderpad=0.0,
        handlelength=1.5,
        handletextpad=0.4,
    )

    # (b) Theorem 3 — use k0 sweep (eps=1e-4) for a clean curve
    ax = axes[1]
    k0_mask = np.array([abs(e - 1e-4) < 1e-12 for e in part_b["tail_eps"]])
    erank_vals = np.array(part_b["erank"])[k0_mask]
    loss_vals = np.array(part_b["constrained_loss"])[k0_mask]
    bound_vals = np.array(part_b["approx_bound"])[k0_mask]
    sort_idx = np.argsort(erank_vals)

    ax.plot(
        erank_vals[sort_idx],
        loss_vals[sort_idx],
        "o-",
        color="#059669",
        markersize=3.5,
        linewidth=1.8,
        label="Constrained loss",
        zorder=3,
    )
    ax.plot(
        erank_vals[sort_idx],
        bound_vals[sort_idx],
        "s--",
        color="#DC2626",
        markersize=3.5,
        linewidth=1.4,
        label="Approx. bound (Thm. 3)",
        zorder=2,
    )
    ax.set_xlabel(r"Effective rank $\mathrm{erank}(H)$")
    ax.set_ylabel(r"Squared loss $/\, m$")
    B = part_b["B_fixed"]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        title=rf"(b) Thm. 3: approximate collapse ($B={B}$)",
        title_fontsize=8.5,
        frameon=False,
        borderpad=0.0,
        handlelength=1.5,
        handletextpad=0.4,
    )

    # (c) erank as proxy — all configs with k* > 0
    ax = axes[2]
    erank_all = np.array(part_b["erank"])
    kstar_all = np.array(part_b["best_k"])
    mask = kstar_all > 0
    ax.scatter(
        erank_all[mask],
        kstar_all[mask],
        color="#2563EB",
        s=30,
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )
    if mask.sum() > 1:
        lim = max(erank_all[mask].max(), kstar_all[mask].max()) + 2
        ax.plot([0, lim], [0, lim], "k:", linewidth=0.8, alpha=0.5, label="$y = x$")
    ax.set_xlabel(r"Effective rank $\mathrm{erank}(H)$")
    ax.set_ylabel(r"Optimal tail index $k^\star$")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        title=r"(c) Remark 4: $\mathrm{erank}$ as tail proxy",
        title_fontsize=8.5,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.0,
    )

    plt.tight_layout(w_pad=1.2)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Running Part A (Theorem 1: exact-rank collapse)...")
    part_a = run_part_a(rng)
    for i, k in enumerate(part_a["ranks"]):
        print(
            f"  rank={k:2d}  loss={part_a['measured_loss'][i]:.6f}  "
            f"bound={part_a['ey_bound'][i]:.6f}  "
            f"erank={part_a['erank'][i]:.2f}"
        )

    print(f"\nRunning Part B (Theorem 3: approximate collapse, B={FIXED_B})...")
    part_b = run_part_b(rng)
    for i in range(len(part_b["k0"])):
        print(
            f"  k0={part_b['k0'][i]:2d}  eps={part_b['tail_eps'][i]:.1e}  "
            f"c_loss={part_b['constrained_loss'][i]:.6f}  "
            f"bound={part_b['approx_bound'][i]:.6f}  "
            f"erank={part_b['erank'][i]:.2f}  "
            f"k*={part_b['best_k'][i]}"
        )

    valid_a = all(
        part_a["measured_loss"][i] >= part_a["ey_bound"][i] - 1e-10
        for i in range(len(part_a["ranks"]))
    )
    print(f"\nTheorem 1 bound valid: {valid_a}")

    valid_b = all(
        part_b["constrained_loss"][i] >= part_b["approx_bound"][i] - 1e-10
        for i in range(len(part_b["k0"]))
    )
    print(f"Theorem 3 bound valid: {valid_b}")

    nonzero_bounds = sum(1 for b in part_b["approx_bound"] if b > 0)
    print(f"Non-trivial (positive) Thm. 3 bounds: {nonzero_bounds}/{len(part_b['k0'])}")

    nonzero_k = [(e, k) for e, k in zip(part_b["erank"], part_b["best_k"]) if k > 0]
    if len(nonzero_k) > 1:
        corr = np.corrcoef([e for e, _ in nonzero_k], [k for _, k in nonzero_k])[0, 1]
        print(f"Correlation(erank, k*) [k*>0]: {corr:.4f} (n={len(nonzero_k)})")

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    make_figure(part_a, part_b, FIG_OUT)

    data_out = FIG_OUT.replace(".pdf", "_data.json")
    with open(data_out, "w") as f:
        json.dump(
            {
                "part_a": part_a,
                "part_b": part_b,
                "dims": DIMS,
                "seed": SEED,
                "B_fixed": FIXED_B,
            },
            f,
            indent=2,
        )
    print(f"Saved data: {data_out}")


if __name__ == "__main__":
    main()
