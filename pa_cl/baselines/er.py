"""Experience Replay (ER) baseline.

Reference: Chaudhry et al., "On tiny episodic memories in continual
learning" (arXiv:1902.10486). At each training step, sample a buffer
batch and concatenate it with the current task minibatch for the
classification loss. Maintain the buffer with reservoir sampling
(Vitter 1985), so it is approximately uniform over all examples seen.

Hyperparameters (Buzzega et al., 2020 / Mammoth defaults):
    buffer_size      : 500 / 2000 / 5120 for Split-CIFAR-100
    batch_size       : training batch (e.g., 32)
    buffer_batch     : usually equal to batch_size; we mix 1:1
    alpha            : weight on buffer cross-entropy (1.0 by default)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReservoirBuffer:
    """Stores (x, y) pairs on GPU; reservoir-sampled add, random sample."""

    def __init__(self, capacity: int, x_shape, device: torch.device,
                 y_dtype: torch.dtype = torch.long):
        self.capacity = int(capacity)
        self.device = device
        self.x = torch.empty((capacity, *x_shape),
                             dtype=torch.float32, device=device)
        self.y = torch.empty((capacity,), dtype=y_dtype, device=device)
        self.seen = 0          # total examples seen
        self.fill = 0          # current buffer occupancy

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        B = x.size(0)
        for i in range(B):
            if self.fill < self.capacity:
                self.x[self.fill] = x[i].detach()
                self.y[self.fill] = y[i].detach()
                self.fill += 1
            else:
                # Reservoir: replace random slot w.p. capacity/(seen+1).
                j = int(torch.randint(0, self.seen + 1, (1,)).item())
                if j < self.capacity:
                    self.x[j] = x[i].detach()
                    self.y[j] = y[i].detach()
            self.seen += 1

    @torch.no_grad()
    def sample(self, batch_size: int) -> Optional:
        if self.fill == 0:
            return None
        idx = torch.randint(0, self.fill, (min(batch_size, self.fill),),
                            device=self.device)
        return self.x.index_select(0, idx), self.y.index_select(0, idx)

    def __len__(self) -> int:
        return self.fill


class ER:
    """Experience Replay: mixes a buffer batch with each task minibatch."""

    def __init__(self, model: nn.Module, buffer_size: int = 500,
                 buffer_batch: int = 32, alpha: float = 1.0,
                 input_shape=None):
        self.model = model
        self.buffer = None       # lazy: init on first batch (so we know shape+device)
        self.buffer_size = int(buffer_size)
        self.buffer_batch = int(buffer_batch)
        self.alpha = float(alpha)
        self.input_shape = tuple(input_shape) if input_shape is not None else None

    def _ensure_buffer(self, ref_tensor: torch.Tensor) -> None:
        if self.buffer is None:
            shape = self.input_shape if self.input_shape is not None \
                else tuple(ref_tensor.shape[1:])
            self.buffer = ReservoirBuffer(
                capacity=self.buffer_size,
                x_shape=shape,
                device=ref_tensor.device,
            )

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._ensure_buffer(x)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        if len(self.buffer) > 0:
            buf = self.buffer.sample(self.buffer_batch)
            if buf is not None:
                bx, by = buf
                bl = self.model(bx)
                loss = loss + self.alpha * F.cross_entropy(bl, by)
        # Stash the current batch so post_step can add it to the buffer
        # *after* the optimizer step, mirroring online reservoir sampling.
        self._last_xy = (x.detach(), y.detach())
        return loss

    def post_step(self) -> None:
        """Called by trainer after optimizer.step(); add current batch to buffer."""
        xy = getattr(self, "_last_xy", None)
        if xy is None or self.buffer is None:
            return
        x, y = xy
        self.buffer.add(x, y)
        self._last_xy = None

    def end_of_task(self, *_args, **_kw) -> None:
        return None