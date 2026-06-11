"""A-GEM baseline (Averaged Gradient Episodic Memory).

Reference: Chaudhry et al. "Efficient lifelong learning with A-GEM"
(ICLR 2019). At each training step, sample a buffer batch and compute
its gradient g_ref; project the current-task gradient g onto the
half-space {v : <v, g_ref> >= 0}. The projection is:
    g_proj = g - (g . g_ref / g_ref . g_ref) * g_ref     if g.g_ref < 0
    g_proj = g                                            otherwise

This guarantees that the parameter update does not, in expectation,
increase the loss on the buffer batch.

Hyperparameters: buffer_size (~500/2000) and buffer_batch (~32).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .er import ReservoirBuffer


class AGEM:
    def __init__(self, model: nn.Module, buffer_size: int = 500,
                 buffer_batch: int = 32, input_shape=None):
        self.model = model
        self.buffer: Optional[ReservoirBuffer] = None
        self.buffer_size = int(buffer_size)
        self.buffer_batch = int(buffer_batch)
        self.input_shape = tuple(input_shape) if input_shape is not None else None

    def _ensure_buffer(self, ref: torch.Tensor) -> None:
        if self.buffer is None:
            shape = self.input_shape if self.input_shape is not None \
                else tuple(ref.shape[1:])
            self.buffer = ReservoirBuffer(
                capacity=self.buffer_size,
                x_shape=shape,
                device=ref.device,
            )

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the *current-task* CE loss only. The projection logic
        operates on gradients post-backward and is invoked by the
        custom `agem_step()` method (see trainer integration)."""
        self._ensure_buffer(x)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self._last_xy = (x.detach(), y.detach())
        return loss

    def project_gradient(self) -> None:
        """If the buffer is non-empty, compute g_ref on a buffer batch,
        flatten both gradients, do the A-GEM projection, and write back."""
        if self.buffer is None or len(self.buffer) == 0:
            return
        # Compute the reference gradient on a buffer sample.
        params = [p for p in self.model.parameters() if p.requires_grad]
        # Snapshot the current (task) gradient.
        g_task = [p.grad.detach().clone() if p.grad is not None
                  else torch.zeros_like(p) for p in params]
        # Compute the buffer-loss gradient on the same model state.
        buf = self.buffer.sample(self.buffer_batch)
        if buf is None:
            return
        bx, by = buf
        self.model.zero_grad(set_to_none=True)
        bl = self.model(bx)
        bf_loss = F.cross_entropy(bl, by)
        bf_loss.backward()
        g_ref = [p.grad.detach().clone() if p.grad is not None
                 else torch.zeros_like(p) for p in params]
        # Flatten and dot.
        flat_t = torch.cat([g.flatten() for g in g_task])
        flat_r = torch.cat([g.flatten() for g in g_ref])
        dot = (flat_t * flat_r).sum()
        if dot < 0:
            norm_r = (flat_r * flat_r).sum().clamp_min(1e-12)
            proj = flat_t - (dot / norm_r) * flat_r
        else:
            proj = flat_t
        # Write back to the params.
        offset = 0
        for p, g_t in zip(params, g_task):
            n = g_t.numel()
            p.grad = proj[offset:offset + n].view_as(g_t).contiguous()
            offset += n

    def post_step(self) -> None:
        xy = getattr(self, "_last_xy", None)
        if xy is None or self.buffer is None:
            return
        x, y = xy
        self.buffer.add(x, y)
        self._last_xy = None

    def end_of_task(self, *_args, **_kw) -> None:
        return None