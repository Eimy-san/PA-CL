"""CLS-ER: Complementary Learning Systems Experience Replay.

Reference: Arani et al., "Learning Fast, Learning Slow: A General
Continual Learning Method based on Complementary Learning System"
(ICLR 2022, arXiv:2201.12604).

CLS-ER maintains two exponential moving average (EMA) copies of the
working model as semantic memories: a plastic model (fast adaptation)
and a stable model (slow consolidation). At each step, the working
model is trained with cross-entropy on the union of stream and buffer
samples, plus a consistency loss (MSE) that aligns the working model's
buffer logits with the semantically richer logits from the EMA models.
After each optimizer step, the EMA models are stochastically updated.

This implementation uses the working model for evaluation (the original
paper uses the stable model) to match the shared-model evaluation
protocol of our training framework.

Hyperparameters (Arani et al. 2022, Table S4):
    buffer_size : 500
    buffer_batch : 32
    alpha_s : stable EMA decay (0.99)
    alpha_p : plastic EMA decay (0.9)
    r_s : stable update rate (0.5)
    r_p : plastic update rate (0.9)
    lmbda : consistency loss weight (0.1)
"""

from __future__ import annotations

import copy
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .er import ReservoirBuffer


class CLSER:
    """CLS-ER: dual EMA semantic memories + reservoir episodic memory."""

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 500,
        buffer_batch: int = 32,
        alpha_s: float = 0.99,
        alpha_p: float = 0.9,
        r_s: float = 0.5,
        r_p: float = 0.9,
        lmbda: float = 0.1,
        input_shape=None,
        num_classes: int = 10,
    ):
        self.model = model
        self.buffer: Optional[ReservoirBuffer] = None
        self.buffer_size = int(buffer_size)
        self.buffer_batch = int(buffer_batch)
        self.alpha_s = float(alpha_s)
        self.alpha_p = float(alpha_p)
        self.r_s = float(r_s)
        self.r_p = float(r_p)
        self.lmbda = float(lmbda)
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.plastic_model: Optional[nn.Module] = None
        self.stable_model: Optional[nn.Module] = None

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

    def _ensure_models(self) -> None:
        if self.plastic_model is None:
            self.plastic_model = copy.deepcopy(self.model)
            self.stable_model = copy.deepcopy(self.model)
            self.plastic_model.train()
            self.stable_model.train()
            for p in self.plastic_model.parameters():
                p.requires_grad_(False)
            for p in self.stable_model.parameters():
                p.requires_grad_(False)

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._ensure_buffer(x)
        self._ensure_models()

        if len(self.buffer) > 0:
            buf = self.buffer.sample(self.buffer_batch)
            if buf is not None:
                bx, by = buf
                combined_x = torch.cat([x, bx], dim=0)
                combined_y = torch.cat([y, by], dim=0)
                logits = self.model(combined_x)
                loss_ce = F.cross_entropy(logits, combined_y)

                with torch.no_grad():
                    z_p = self.plastic_model(bx)
                    z_s = self.stable_model(bx)
                    p_p = F.softmax(z_p, dim=1)
                    p_s = F.softmax(z_s, dim=1)
                    pp_true = p_p.gather(1, by.unsqueeze(1))
                    ps_true = p_s.gather(1, by.unsqueeze(1))
                    use_plastic = pp_true > ps_true
                    z_selected = torch.where(use_plastic, z_p, z_s)

                buf_logits = logits[x.size(0) :]
                loss_cons = self.lmbda * F.mse_loss(buf_logits, z_selected)
                loss = loss_ce + loss_cons
            else:
                loss = F.cross_entropy(self.model(x), y)
        else:
            loss = F.cross_entropy(self.model(x), y)

        self._last_xy = (x.detach(), y.detach())
        return loss

    def post_step(self) -> None:
        xy = getattr(self, "_last_xy", None)
        if xy is not None and self.buffer is not None:
            x, y = xy
            self.buffer.add(x, y)
            self._last_xy = None

        if self.plastic_model is not None:
            if random.random() < self.r_p:
                self._update_ema(self.plastic_model, self.alpha_p)
            if random.random() < self.r_s:
                self._update_ema(self.stable_model, self.alpha_s)

    def _update_ema(self, ema_model: nn.Module, alpha: float) -> None:
        with torch.no_grad():
            for ep, mp in zip(ema_model.parameters(), self.model.parameters()):
                ep.mul_(alpha).add_(mp.data, alpha=(1 - alpha))
            for eb, mb in zip(ema_model.buffers(), self.model.buffers()):
                eb.copy_(mb)

    def end_of_task(self, *_args, **_kw) -> None:
        return None
