"""Reference implementation of PA-CL.

The PACL class is a *wrapper* around any base continual learner. It adds:
  1. an effective-rank floor tracked per regularized layer
  2. a one-sided squared-hinge effective-rank loss
  3. a diagonal Fisher accumulator (single, exponentially weighted)
  4. a Fisher-null-space projection of the rank-loss gradient

The class exposes:
  - register_hooks(model, layer_names): attach forward hooks to capture
    activations from the regularized layers
  - rank_loss(): compute the per-step rank loss from the captured
    activations (called before backward)
  - project_rank_grad(model): apply Fisher-null projection to the
    rank-loss gradient that is currently stored in model.parameters().grad
  - end_of_task(model, loader, device): update Fisher accumulator and
    rank floors at the end of each task

This module is deliberately self-contained and dependency-light:
only torch + numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------
# Effective rank
# -----------------------------------------------------------------

def effective_rank(H: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Effective rank of a (B, D) activation matrix.

    Defined as exp(spectral entropy of singular value distribution).
    Differentiable wherever the spectrum has no exact repeated values.

    Implementation note: we use torch.linalg.svdvals which is
    autograd-aware on modern PyTorch.
    """
    if H.dim() != 2:
        H = H.reshape(H.shape[0], -1)
    # Center the activations to remove the constant component.
    H = H - H.mean(dim=0, keepdim=True)
    # Singular values (already non-negative).
    sigma = torch.linalg.svdvals(H)
    sigma = sigma + eps
    p = sigma / sigma.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy)


# -----------------------------------------------------------------
# PA-CL state container
# -----------------------------------------------------------------

@dataclass
class PACLConfig:
    """Hyperparameters with defaults from paper Sec. 4."""
    lam: float = 0.1                    # rank-loss weight  (lambda)
    beta: float = 0.99                  # EMA for rank floor update
    alpha: float = 0.9                  # Fisher accumulator decay
    tau: float = 1e-4                   # Fisher-null projection floor
    layer_weights: Optional[Dict[str, float]] = None  # per-layer w_l
    # Ablation flag: when True, project_rank_grad is a no-op (returns
    # the rank gradient unchanged). Used in Section 6.2 to isolate the
    # contribution of the Fisher-null projection to BWT.
    disable_projection: bool = False


class PACL:
    """Plasticity-Aware Continual Learning wrapper."""

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        config: PACLConfig | None = None,
    ):
        self.model = model
        self.layer_names = list(layer_names)
        self.cfg = config or PACLConfig()
        if self.cfg.layer_weights is None:
            self.cfg.layer_weights = {n: 1.0 for n in self.layer_names}

        # Activation cache: layer_name -> Tensor (B, D)
        self._cache: Dict[str, torch.Tensor] = {}
        # Per-layer running rank floor (Python float, monotone non-dec)
        self.rank_floor: Dict[str, float] = {n: 0.0 for n in self.layer_names}
        # Per-layer EMA of current-batch rank, used to update the floor
        self._ema_rank: Dict[str, float] = {n: 0.0 for n in self.layer_names}
        # Fisher accumulator: parameter name -> Tensor
        self.fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    # ----- hook plumbing -----

    def _register_hooks(self) -> None:
        modules = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in modules:
                raise KeyError(
                    f"Layer '{name}' not found in model. "
                    f"Available: {list(modules.keys())[:20]}..."
                )
            modules[name].register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def _hook(_module, _inp, out):
            # Cache the post-activation output, kept on its device.
            if isinstance(out, tuple):
                out = out[0]
            self._cache[name] = out
        return _hook

    def detach_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ----- rank loss -----

    def rank_loss(self) -> torch.Tensor:
        """One-sided squared hinge on the rank floor, summed across layers."""
        if not self._cache:
            return torch.zeros((), device=next(self.model.parameters()).device)
        device = next(iter(self._cache.values())).device
        loss = torch.zeros((), device=device)
        layer_weights = self.cfg.layer_weights or {}
        for name, H in self._cache.items():
            w = layer_weights.get(name, 1.0)
            rho = effective_rank(H)
            deficit = torch.clamp(self.rank_floor[name] - rho, min=0.0)
            loss = loss + w * deficit.pow(2)
            # Track EMA of the *un-thresholded* rank for floor updates.
            r = float(rho.detach().cpu())
            self._ema_rank[name] = (
                self.cfg.beta * self._ema_rank[name]
                + (1.0 - self.cfg.beta) * r
            )
        # Clear cache so hooks repopulate next forward.
        self._cache.clear()
        return loss

    # ----- Fisher-null projection -----

    def project_rank_grad(
        self,
        rank_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply M = F / (F + tau) attenuation; return (1 - M) * g_rank.

        Operates per-parameter, element-wise. If
        `self.cfg.disable_projection` is True, returns the rank
        gradient unchanged (Section 6.2 ablation: isolates the
        contribution of the Fisher-null projection to BWT).
        """
        if self.cfg.disable_projection:
            return dict(rank_grads)
        out: Dict[str, torch.Tensor] = {}
        for n, g in rank_grads.items():
            F_n = self.fisher[n]
            M = F_n / (F_n + self.cfg.tau)
            out[n] = (1.0 - M) * g
        return out

    # ----- end-of-task housekeeping -----

    @torch.no_grad()
    def update_rank_floor(self) -> None:
        """Take the per-layer max(floor, EMA-rank) update."""
        for name in self.layer_names:
            self.rank_floor[name] = max(
                self.rank_floor[name],
                float(self._ema_rank[name]),
            )

    def update_fisher(
        self,
        loader: Iterable,
        device: torch.device,
        n_samples: int = 2000,
    ) -> None:
        """Update the running diagonal Fisher: F_t = alpha * F_{t-1} + F_hat_t.

        F_hat_t is estimated by squaring the per-sample gradient of
        log p(y|x) on up to `n_samples` examples drawn from `loader`.
        """
        self.model.eval()
        for n, _ in self.model.named_parameters():
            if n in self.fisher:
                self.fisher[n].mul_(self.cfg.alpha)
        fisher_accum = {n: torch.zeros_like(p) for n, p in
                        self.model.named_parameters() if p.requires_grad}
        seen = 0
        n_batches = 0
        last_batch_size = 1  # safe default for the denominator
        for batch in loader:
            if seen >= n_samples:
                break
            x = batch[0].to(device)
            y = batch[1].to(device)
            last_batch_size = max(1, x.size(0))
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            log_p = F.log_softmax(logits, dim=-1)
            # Negative log-likelihood; averaged here, sampled-style Fisher
            # estimator on a per-batch basis.
            sampled = log_p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            nll = -sampled.mean()
            nll.backward()
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                fisher_accum[n].add_(p.grad.detach().pow(2))
            seen += last_batch_size
            n_batches += 1
        # Average over the number of batches actually used.
        denom = max(1, n_batches)
        for n, p in self.model.named_parameters():
            if n not in self.fisher:
                continue
            self.fisher[n].add_(fisher_accum[n] / denom)
        self.model.zero_grad(set_to_none=True)