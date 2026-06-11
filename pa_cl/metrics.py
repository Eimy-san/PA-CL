"""Continual-learning metrics.

Two complementary protocols are supported:

(1) Full O(T^2) protocol: a T x T accuracy matrix R is recorded
    (R[i, j] = test acc on task j after training on tasks 0..i). This
    is the standard Lopez-Paz & Ranzato (2017) protocol used in
    short-horizon Class-IL benchmarks.

(2) O(T) sparse protocol: only the diagonal `diag_acc` (test acc on
    task t immediately after training task t) and the final row
    `final_row` (test acc on every task after the full stream is done)
    are recorded. From these two we can compute ACC, an end-of-stream
    BWT, the plasticity-decay curve, and per-task forgetting. This is
    the protocol used by Dohare et al. (2024, Nature) for long-horizon
    Continual-Permuted-MNIST runs (T >= 200), where the O(T^2) version
    is computationally infeasible.

Both protocols are exposed below.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np


# -----------------------------------------------------------------
# Full O(T^2) protocol
# -----------------------------------------------------------------

def acc(R: np.ndarray) -> float:
    """Average accuracy: mean of last row of the accuracy matrix R."""
    return float(np.mean(R[-1, : R.shape[0]]))


def bwt(R: np.ndarray) -> float:
    """Backward transfer over all past tasks (Lopez-Paz & Ranzato, 2017)."""
    T = R.shape[0]
    if T < 2:
        return 0.0
    diffs = [R[-1, t] - R[t, t] for t in range(T - 1)]
    return float(np.mean(diffs))


def fwt(R: np.ndarray, rand_baseline: Optional[np.ndarray] = None) -> float:
    """Forward transfer; needs baseline accuracy of random init per task."""
    T = R.shape[0]
    if T < 2 or rand_baseline is None:
        return 0.0
    diffs = [R[t - 1, t] - rand_baseline[t] for t in range(1, T)]
    return float(np.mean(diffs))


def plasticity_curve(R: np.ndarray) -> List[float]:
    """Fresh-task accuracies a_{t,t}, the diagonal of R."""
    return [float(R[t, t]) for t in range(R.shape[0])]


def per_task_forgetting(R: np.ndarray) -> List[float]:
    """f_t = max_{s <= t} R[s, t] - R[-1, t]."""
    T = R.shape[0]
    out = []
    for t in range(T - 1):
        best = max(R[s, t] for s in range(t, T - 1))
        out.append(float(best - R[-1, t]))
    return out


# -----------------------------------------------------------------
# Sparse O(T) protocol (diag_acc + final_row)
# -----------------------------------------------------------------

def acc_sparse(final_row: Sequence[float]) -> float:
    """Average accuracy over all tasks at end of stream."""
    a = np.asarray(final_row, dtype=float)
    return float(a.mean()) if a.size > 0 else 0.0


def bwt_sparse(diag_acc: Sequence[float], final_row: Sequence[float]) -> float:
    """End-of-stream backward transfer = mean over s<T of (final[s] - diag[s])."""
    d = np.asarray(diag_acc, dtype=float)
    f = np.asarray(final_row, dtype=float)
    T = min(d.shape[0], f.shape[0])
    if T < 2:
        return 0.0
    return float((f[:T - 1] - d[:T - 1]).mean())


def plasticity_decay(diag_acc: Sequence[float]) -> List[float]:
    """The diagonal sequence itself is the plasticity-decay signal."""
    return [float(v) for v in diag_acc]


def per_task_forgetting_sparse(
    diag_acc: Sequence[float], final_row: Sequence[float]
) -> List[float]:
    """f_t = diag[t] - final[t] for t < T-1."""
    d = np.asarray(diag_acc, dtype=float)
    f = np.asarray(final_row, dtype=float)
    T = min(d.shape[0], f.shape[0])
    return [float(d[t] - f[t]) for t in range(T - 1)]


# -----------------------------------------------------------------
# Effective rank (numpy/torch agnostic helper)
# -----------------------------------------------------------------

def effective_rank_probe(H, eps: float = 1e-12) -> float:
    """Effective rank of a (B, D) activation matrix as a python float.

    NaN/Inf-safe: returns 0.0 if the input contains non-finite values
    (can happen during early training when the model produces degenerate
    activations).
    """
    try:
        import torch
        if isinstance(H, torch.Tensor):
            with torch.no_grad():
                if H.dim() != 2:
                    H = H.reshape(H.shape[0], -1)
                if not torch.isfinite(H).all():
                    return 0.0
                H = H - H.mean(dim=0, keepdim=True)
                sigma = torch.linalg.svdvals(H).cpu().numpy()
        else:
            sigma = None
    except ImportError:
        sigma = None
    if sigma is None:
        H_np = np.asarray(H)
        if H_np.ndim != 2:
            H_np = H_np.reshape(H_np.shape[0], -1)
        if not np.isfinite(H_np).all():
            return 0.0
        H_np = H_np - H_np.mean(axis=0, keepdims=True)
        sigma = np.linalg.svd(H_np, compute_uv=False)
    sigma = sigma + eps
    if sigma.sum() < eps:
        return 0.0
    p = sigma / sigma.sum()
    entropy = float(-(p * np.log(p + eps)).sum())
    return float(np.exp(entropy))