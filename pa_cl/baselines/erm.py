"""ERM baseline: plain SGD on the current task, no past-task machinery.

This is the minimal sanity baseline. All other baselines (EWC, ER,
DER++, GPM, CBP, ...) will subclass / compose with this one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ERM:
    """Standard cross-entropy training. Stateless across tasks."""

    def __init__(self, model: nn.Module):
        self.model = model

    def task_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(x)
        return F.cross_entropy(logits, y)

    def end_of_task(self, *_args, **_kw) -> None:
        """No-op: ERM does not track any per-task state."""
        return None