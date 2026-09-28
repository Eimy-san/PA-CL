"""Experience Replay with Asymmetric Cross-Entropy (ER-ACE).

Reference: Caccia et al., "New Insights on Reducing Abrupt Representation
Change in Online Continual Learning" (ICLR 2022, arXiv:2104.05025).

ER-ACE shields previously learned class representations from disruption
by new data. On each incoming batch, the cross-entropy loss masks logits
for classes that have been seen but are not present in the current batch,
forcing new classes to adapt to old representations rather than the
reverse. Buffer samples are trained with standard cross-entropy.

In domain-incremental settings where all classes appear in every task
(e.g., Permuted MNIST), the asymmetric mask has no effect after the first
task, and ER-ACE reduces to standard ER.

Hyperparameters (Caccia et al. 2022 / Mammoth defaults):
    buffer_size : 500
    buffer_batch : 32
    alpha : weight on buffer CE (1.0)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .er import ReservoirBuffer


class ERAce:
    """ER with asymmetric cross-entropy on the stream, standard CE on buffer."""

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 500,
        buffer_batch: int = 32,
        alpha: float = 1.0,
        num_classes: int = 10,
        input_shape=None,
    ):
        self.model = model
        self.buffer: Optional[ReservoirBuffer] = None
        self.buffer_size = int(buffer_size)
        self.buffer_batch = int(buffer_batch)
        self.alpha = float(alpha)
        self.num_classes = int(num_classes)
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.seen_so_far = torch.zeros(self.num_classes, dtype=torch.bool)

    def _ensure_buffer(self, ref: torch.Tensor) -> None:
        if self.buffer is None:
            shape = (
                self.input_shape
                if self.input_shape is not None
                else tuple(ref.shape[1:])
            )
            self.buffer = ReservoirBuffer(
                capacity=self.buffer_size,
                x_shape=shape,
                device=ref.device,
            )

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._ensure_buffer(x)
        self.seen_so_far = self.seen_so_far.to(x.device)

        present = y.unique()
        self.seen_so_far[present] = True

        logits = self.model(x)

        seen_classes = torch.where(self.seen_so_far)[0]
        mask = torch.zeros_like(logits)
        mask[:, present] = 1.0
        if seen_classes.numel() > 0:
            max_seen = int(seen_classes.max().item())
            if max_seen + 1 < logits.size(1):
                mask[:, max_seen + 1 :] = 1.0

        if mask.sum() < mask.numel():
            masked_logits = logits.masked_fill(mask == 0, -1e9)
            loss = F.cross_entropy(masked_logits, y)
        else:
            loss = F.cross_entropy(logits, y)

        if len(self.buffer) > 0:
            buf = self.buffer.sample(self.buffer_batch)
            if buf is not None:
                bx, by = buf
                bl = self.model(bx)
                loss = loss + self.alpha * F.cross_entropy(bl, by)

        self._last_xy = (x.detach(), y.detach())
        return loss

    def post_step(self) -> None:
        xy = getattr(self, "_last_xy", None)
        if xy is None or self.buffer is None:
            return
        x, y = xy
        self.buffer.add(x, y)
        self._last_xy = None

    def end_of_task(self, *_args, **_kw) -> None:
        return None
