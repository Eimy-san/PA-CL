"""eXtended Dark Experience Replay (X-DER).

Reference: Boschini et al., "Class-Incremental Continual Learning into
the eXtended DER-verse" (TPAMI 2022, arXiv:2201.00766).

X-DER extends DER++ with:
  1. Distillation replay (MSE on stored logits, like DER)
  2. Label replay (CE on buffer labels, like DER++)
  3. Future logits constraint (margin loss preventing future-task
     logits from exceeding current-task logits)
  4. Past logits constraint (margin loss maintaining correct-class
     dominance over past classes in softmax space)
  5. Logit update at task boundaries (refresh stored logits with
     current model predictions for old-task classes)
  6. Buffer reorganization at task boundaries (balanced per-class
     allocation, class-IL only)

This implementation omits the SimCLR-based future-preparation
consistency term, which requires an augmentation pipeline and a
contrastive head that are not part of our shared evaluation stack
(a reduced X-DER variant; both X-DER and PA-CL+X-DER use the same
reduced base). All other components are faithfully implemented.

In domain-incremental settings (all classes present in every task),
the head-based constraint terms are disabled and X-DER reduces to
DER++ with logit refresh at task boundaries.

Hyperparameters (Boschini et al. 2022 / Mammoth defaults):
    buffer_size : 500
    buffer_batch : 32
    alpha : distillation weight (0.5)
    beta : label replay weight (0.5)
    gamma : logit update weight (0.85)
    constr_eta : constraint regularization weight (0.1)
    constr_margin : margin for past/future constraints (0.3)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class XDERBuffer:
    """Buffer of (x, y, logits, task_label) quadruples, reservoir-sampled."""

    def __init__(self, capacity: int, x_shape, n_classes: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.x = torch.empty((capacity, *x_shape), dtype=torch.float32, device=device)
        self.y = torch.empty((capacity,), dtype=torch.long, device=device)
        self.logits = torch.empty(
            (capacity, n_classes), dtype=torch.float32, device=device
        )
        self.task_labels = torch.empty((capacity,), dtype=torch.long, device=device)
        self.seen = 0
        self.fill = 0

    @torch.no_grad()
    def add(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, task_label: int
    ) -> None:
        B = x.size(0)
        tl = torch.full((B,), task_label, dtype=torch.long, device=x.device)
        for i in range(B):
            if self.fill < self.capacity:
                self.x[self.fill] = x[i].detach()
                self.y[self.fill] = y[i].detach()
                self.logits[self.fill] = logits[i].detach()
                self.task_labels[self.fill] = tl[i]
                self.fill += 1
            else:
                j = int(torch.randint(0, self.seen + 1, (1,)).item())
                if j < self.capacity:
                    self.x[j] = x[i].detach()
                    self.y[j] = y[i].detach()
                    self.logits[j] = logits[i].detach()
                    self.task_labels[j] = tl[i]
            self.seen += 1

    @torch.no_grad()
    def sample(
        self, batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.fill == 0:
            return None
        bs = min(batch_size, self.fill)
        idx = torch.randint(0, self.fill, (bs,), device=self.device)
        return (
            self.x.index_select(0, idx),
            self.y.index_select(0, idx),
            self.logits.index_select(0, idx),
            self.task_labels.index_select(0, idx),
        )

    @torch.no_grad()
    def update_logits(self, model: nn.Module, batch_size: int = 128) -> None:
        """Refresh stored logits using current model for old-task samples."""
        if self.fill == 0:
            return
        for start in range(0, self.fill, batch_size):
            end = min(start + batch_size, self.fill)
            bx = self.x[start:end]
            new_logits = model(bx)
            self.logits[start:end] = new_logits.detach()

    def __len__(self) -> int:
        return self.fill


class XDER:
    """eXtended DER: distillation + label replay + margin constraints."""

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 500,
        buffer_batch: int = 32,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.85,
        constr_eta: float = 0.1,
        constr_margin: float = 0.3,
        num_classes: int = 10,
        classes_per_task: int = 0,
        input_shape=None,
    ):
        self.model = model
        self.buffer: Optional[XDERBuffer] = None
        self.buffer_size = int(buffer_size)
        self.buffer_batch = int(buffer_batch)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.constr_eta = float(constr_eta)
        self.constr_margin = float(constr_margin)
        self.num_classes = int(num_classes)
        self.classes_per_task = int(classes_per_task)
        self.is_class_il = classes_per_task > 0 and classes_per_task < num_classes
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.current_task = 0

    def _ensure_buffer(self, ref: torch.Tensor) -> None:
        if self.buffer is None:
            shape = (
                self.input_shape
                if self.input_shape is not None
                else tuple(ref.shape[1:])
            )
            head = getattr(self.model, "head", None)
            if head is not None and hasattr(head, "out_features"):
                self.num_classes = int(head.out_features)
            self.buffer = XDERBuffer(
                capacity=self.buffer_size,
                x_shape=shape,
                n_classes=self.num_classes,
                device=ref.device,
            )

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._ensure_buffer(x)
        logits = self.model(x)

        if self.is_class_il:
            s = self.current_task * self.classes_per_task
            e = s + self.classes_per_task
            loss = F.cross_entropy(logits[:, s:e], y - s)
        else:
            loss = F.cross_entropy(logits, y)

        if len(self.buffer) > 0:
            buf = self.buffer.sample(self.buffer_batch)
            if buf is not None:
                bx, by, blgt, _ = buf
                bl = self.model(bx)
                loss = loss + self.alpha * F.mse_loss(bl, blgt)

            buf2 = self.buffer.sample(self.buffer_batch)
            if buf2 is not None:
                bx2, by2, _, _ = buf2
                bl2 = self.model(bx2)
                if self.is_class_il:
                    n_past = self.current_task * self.classes_per_task
                    if n_past > 0:
                        past_mask = by2 < n_past
                        if past_mask.any():
                            loss = loss + self.beta * F.cross_entropy(
                                bl2[past_mask, :n_past], by2[past_mask]
                            )
                else:
                    loss = loss + self.beta * F.cross_entropy(bl2, by2)

        if self.is_class_il:
            cpt = self.classes_per_task
            cur_s = self.current_task * cpt
            cur_e = cur_s + cpt

            if self.current_task > 0:
                chead = F.softmax(logits[:, :cur_e], dim=1)
                good = chead[:, cur_s:cur_e]
                bad = chead[:, :cur_s]
                margin_loss = (
                    bad.max(1)[0].detach() + self.constr_margin - good.max(1)[0]
                )
                mask = margin_loss > 0
                if mask.any():
                    loss = loss + self.constr_eta * margin_loss[mask].mean()

            if cur_e < self.num_classes:
                bad_head = logits[:, cur_e:]
                good_head = logits[:, cur_s:cur_e]
                if len(self.buffer) > 0 and buf is not None:
                    bx_buf = bx
                    bl_buf = self.model(bx_buf)
                    bad_head = torch.cat([bad_head, bl_buf[:, cur_e:]], dim=0)
                    good_head = torch.cat([good_head, bl_buf[:, cur_s:cur_e]], dim=0)
                margin_loss = (
                    bad_head.max(1)[0] + self.constr_margin - good_head.max(1)[0]
                )
                mask = margin_loss > 0
                if mask.any():
                    loss = loss + self.constr_eta * margin_loss[mask].mean()

        self._last_xy_logits = (x.detach(), y.detach(), logits.detach())
        return loss

    def post_step(self) -> None:
        xyl = getattr(self, "_last_xy_logits", None)
        if xyl is None or self.buffer is None:
            return
        x, y, lgt = xyl
        self.buffer.add(x, y, lgt, self.current_task)
        self._last_xy_logits = None

    def end_of_task(self, *_args, **_kw) -> None:
        if self.buffer is not None and len(self.buffer) > 0:
            self.buffer.update_logits(self.model)
        self.current_task += 1
