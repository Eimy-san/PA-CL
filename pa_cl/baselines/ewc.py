"""Online EWC baseline.

Reference: Schwarz et al. "Progress & Compress" (ICML 2018), the
"online" variant of Kirkpatrick et al. 2017 EWC. After each task we
update a single running diagonal Fisher with geometric weighting:
    F_t = gamma * F_{t-1} + F_hat_t
and snapshot the post-task parameters as theta_t^*. The CL loss is:
    L_task + (lambda / 2) * sum_p F_p * (theta_p - theta_p^*)^2

Hyperparameters (Buzzega et al. 2020, Mammoth defaults):
    lambda_ewc : 10 for Split-CIFAR-100
    gamma      : 1.0 (no decay; just sum the Fishers)
    fisher_samples : 1000 (per-task sample budget for Fisher estimation)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    def __init__(self, model: nn.Module, lambda_ewc: float = 10.0,
                 gamma: float = 1.0, fisher_samples: int = 1000):
        self.model = model
        self.lambda_ewc = float(lambda_ewc)
        self.gamma = float(gamma)
        self.fisher_samples = int(fisher_samples)
        self.fisher: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}

    def _init_state(self) -> None:
        if not self.fisher:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.fisher[n] = torch.zeros_like(p)
                    self.theta_star[n] = p.detach().clone()

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._init_state()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        if self.lambda_ewc > 0:
            penalty = torch.zeros((), device=x.device)
            for n, p in self.model.named_parameters():
                if n not in self.fisher:
                    continue
                penalty = penalty + (self.fisher[n] * (p - self.theta_star[n]).pow(2)).sum()
            loss = loss + 0.5 * self.lambda_ewc * penalty
        return loss

    def end_of_task(self, *, train_iter_fn=None, device=None,
                    **_kw) -> None:
        """Update the running Fisher and the parameter snapshot.

        `train_iter_fn` is a callable that yields fresh (x, y) batches
        from the just-finished task; the trainer passes it as a closure
        over the task's `.train_iter(batch_size)` method.
        """
        if train_iter_fn is None:
            # Just snapshot and zero Fisher contribution from this task.
            for n, p in self.model.named_parameters():
                if n in self.theta_star:
                    self.theta_star[n] = p.detach().clone()
            return

        # Estimate empirical Fisher (sampled-label likelihood).
        self.model.eval()
        f_accum = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                   if p.requires_grad}
        seen = 0
        n_batches = 0
        for x, y in train_iter_fn():
            if seen >= self.fisher_samples:
                break
            x = x.to(device) if device is not None else x
            y = y.to(device) if device is not None else y
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            log_p = F.log_softmax(logits, dim=-1)
            sampled = log_p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            nll = -sampled.mean()
            nll.backward()
            for n, p in self.model.named_parameters():
                if p.grad is None or n not in f_accum:
                    continue
                f_accum[n].add_(p.grad.detach().pow(2))
            seen += x.size(0)
            n_batches += 1
        denom = max(1, n_batches)
        for n, p in self.model.named_parameters():
            if n not in self.fisher:
                continue
            self.fisher[n] = self.gamma * self.fisher[n] + f_accum[n] / denom
            self.theta_star[n] = p.detach().clone()
        self.model.zero_grad(set_to_none=True)