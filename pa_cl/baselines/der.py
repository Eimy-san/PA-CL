"""DER++ (Dark Experience Replay++) baseline.

Reference: Buzzega et al. "Dark Experience for General Continual
Learning: a Strong, Simple Baseline" (NeurIPS 2020). DER++ stores
(x, y, logits) triples in a reservoir buffer and uses TWO replay losses:
  1. alpha * MSE(model(x_buf), stored_logits_buf)       (dark replay)
  2. beta  * CrossEntropy(model(x_buf), y_buf)          (++ extension)

The first term distills the soft prediction made by the model at the
time of storage, which is empirically a stronger anti-forgetting
signal than the hard label alone.

Hyperparameters (Buzzega et al. 2020, Table 4 for Split-CIFAR-100):
    buffer_size : 500 / 2000 / 5120
    alpha       : 0.5 (distillation weight)
    beta        : 0.5 (++ CE weight)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DERBuffer:
    """Buffer of (x, y, logits) triples on GPU, reservoir-sampled."""

    def __init__(self, capacity: int, x_shape, n_classes: int,
                 device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.x = torch.empty((capacity, *x_shape), dtype=torch.float32, device=device)
        self.y = torch.empty((capacity,), dtype=torch.long, device=device)
        self.logits = torch.empty((capacity, n_classes),
                                  dtype=torch.float32, device=device)
        self.seen = 0
        self.fill = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> None:
        B = x.size(0)
        for i in range(B):
            if self.fill < self.capacity:
                self.x[self.fill] = x[i].detach()
                self.y[self.fill] = y[i].detach()
                self.logits[self.fill] = logits[i].detach()
                self.fill += 1
            else:
                j = int(torch.randint(0, self.seen + 1, (1,)).item())
                if j < self.capacity:
                    self.x[j] = x[i].detach()
                    self.y[j] = y[i].detach()
                    self.logits[j] = logits[i].detach()
            self.seen += 1

    @torch.no_grad()
    def sample(self, batch_size: int
               ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.fill == 0:
            return None
        bs = min(batch_size, self.fill)
        idx = torch.randint(0, self.fill, (bs,), device=self.device)
        return (self.x.index_select(0, idx),
                self.y.index_select(0, idx),
                self.logits.index_select(0, idx))

    def __len__(self) -> int:
        return self.fill


class DERpp:
    """DER++ wrapper. Mixes (x, y, logits) buffer samples with the current batch."""

    def __init__(self, model: nn.Module, buffer_size: int = 500,
                 buffer_batch: int = 32, alpha: float = 0.5, beta: float = 0.5,
                 input_shape=None, num_classes: int = 100):
        self.model = model
        self.buffer: Optional[DERBuffer] = None
        self.buffer_size = int(buffer_size)
        self.buffer_batch = int(buffer_batch)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.num_classes = int(num_classes)

    def _ensure_buffer(self, ref: torch.Tensor) -> None:
        if self.buffer is None:
            shape = self.input_shape if self.input_shape is not None \
                else tuple(ref.shape[1:])
            # Infer num_classes from the model's last linear layer if possible.
            head = getattr(self.model, "head", None)
            if head is not None and hasattr(head, "out_features"):
                self.num_classes = int(head.out_features)
            self.buffer = DERBuffer(
                capacity=self.buffer_size, x_shape=shape,
                n_classes=self.num_classes, device=ref.device,
            )

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._ensure_buffer(x)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        if len(self.buffer) > 0:
            # Dark replay: MSE on stored logits.
            buf = self.buffer.sample(self.buffer_batch)
            if buf is not None:
                bx, by, blgt = buf
                bl = self.model(bx)
                loss = loss + self.alpha * F.mse_loss(bl, blgt)
                # ++ : CE on stored labels (use a second buffer sample to
                # mirror the published implementation).
                buf2 = self.buffer.sample(self.buffer_batch)
                if buf2 is not None:
                    bx2, by2, _ = buf2
                    bl2 = self.model(bx2)
                    loss = loss + self.beta * F.cross_entropy(bl2, by2)
        # Stash current batch and its logits for post_step.
        self._last_xy_logits = (x.detach(), y.detach(), logits.detach())
        return loss

    def post_step(self) -> None:
        xyl = getattr(self, "_last_xy_logits", None)
        if xyl is None or self.buffer is None:
            return
        x, y, lgt = xyl
        self.buffer.add(x, y, lgt)
        self._last_xy_logits = None

    def end_of_task(self, *_args, **_kw) -> None:
        return None