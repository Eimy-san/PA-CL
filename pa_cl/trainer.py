"""Continual training loop with optional PA-CL wrapping.

The trainer accepts either:
  (a) a list of `PermutedTask` objects (the new fast GPU-resident
      benchmark interface), or
  (b) a list of `torch.utils.data.DataLoader` objects (legacy).

For both, the same training loop applies:
  for t in 1..T:
    for epoch in 1..E:
      for batch in train_iter(t):
        compute task loss, take a step
        if PA-CL: combined-gradient step with Fisher-weighted attenuation
        if base.post_step exists: call it (CBP)
    diag_acc[t] = test accuracy on task t   (plasticity signal)
    rank_traj[t] = per-layer effective rank on a probe batch
    base.end_of_task() ; if PA-CL: update rank floor + Fisher
  final_row = test accuracy on ALL tasks after the stream ends

This O(T) eval schedule is the standard CPMNIST protocol used by
Dohare et al. (2024 Nature) for long-horizon runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .metrics import effective_rank_probe


@dataclass
class TrainerResult:
    diag_acc: list = field(default_factory=list)  # length T, plasticity signal
    final_row: list = field(default_factory=list)  # R[-1, :], length T
    rank_traj: list = field(default_factory=list)  # length T, dict layer->float
    rank_floor_traj: list = field(default_factory=list)  # length T, only for PA-CL
    train_loss_traj: list = field(default_factory=list)  # mean task-loss per task
    seconds: float = 0.0


# ---------------------------------------------------------------
# Unified iter interface: tasks support .train_iter / .test_iter /
# .probe_batch; DataLoaders need an adapter.
# ---------------------------------------------------------------


def _train_iter(task_or_loader, batch_size: int):
    if hasattr(task_or_loader, "train_iter"):
        return task_or_loader.train_iter(batch_size, shuffle=True)
    return iter(task_or_loader)


def _test_iter(task_or_loader, batch_size: int = 2048):
    if hasattr(task_or_loader, "test_iter"):
        return task_or_loader.test_iter(batch_size)
    return iter(task_or_loader)


def _probe_batch(task_or_loader, device: torch.device, n: int = 512):
    if hasattr(task_or_loader, "probe_batch"):
        x, y = task_or_loader.probe_batch(n=n)
        return x.to(device), y.to(device)
    # DataLoader fallback: concat batches until we have n samples.
    xs, ys = [], []
    have = 0
    for x, y in task_or_loader:
        xs.append(x)
        ys.append(y)
        have += x.shape[0]
        if have >= n:
            break
    x = torch.cat(xs, dim=0)[:n].to(device)
    y = torch.cat(ys, dim=0)[:n].to(device)
    return x, y


def evaluate(
    model: nn.Module,
    task_or_loader,
    device: torch.device,
    batch_size: int = 2048,
    max_batches: int = -1,
) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(_test_iter(task_or_loader, batch_size)):
            if max_batches > 0 and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def train_continual(
    base,
    pacl,
    train_loaders,  # may be list[PermutedTask] OR list[DataLoader]
    test_loaders,  # same
    optimizer,
    device: torch.device,
    epochs_per_task: int = 1,
    log_every: int = 100,
    probe_layer_names: Optional[List[str]] = None,
    batch_size: int = 16,
) -> TrainerResult:
    """Train a continual learner across the task stream."""
    model: nn.Module = base.model
    T = len(train_loaders)

    layers_for_probe: List[str] = []
    if pacl is not None:
        layers_for_probe = list(pacl.layer_names)
    elif probe_layer_names is not None:
        layers_for_probe = list(probe_layer_names)

    # Probe hooks for non-PA-CL runs (PA-CL already hooks them).
    probe_cache: Dict[str, torch.Tensor] = {}
    probe_handles: list = []
    if pacl is None and layers_for_probe:
        modules = dict(model.named_modules())
        for name in layers_for_probe:
            if name not in modules:
                raise KeyError(f"probe layer '{name}' not in model")

            def _make(nm):
                def _h(_m, _i, out):
                    probe_cache[nm] = out

                return _h

            probe_handles.append(modules[name].register_forward_hook(_make(name)))

    diag_acc: List[float] = []
    rank_traj: List[Dict[str, float]] = []
    rank_floor_traj: List[Dict[str, float]] = []
    train_loss_traj: List[float] = []
    t0 = time.time()

    for t in range(T):
        task = train_loaders[t]
        running_loss = 0.0
        running_n = 0
        for ep in range(epochs_per_task):
            model.train()
            for it, (x, y) in enumerate(_train_iter(task, batch_size)):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                loss_task = base.task_loss(x, y)
                running_loss += float(loss_task.detach()) * x.size(0)
                running_n += x.size(0)

                if pacl is None:
                    loss_task.backward()
                    # AGEM-style gradient projection (a no-op if buffer empty).
                    if hasattr(base, "project_gradient"):
                        base.project_gradient()
                    optimizer.step()
                else:
                    # Single-forward, two-backward path:
                    # The previous forward (inside task_loss) already
                    # filled pacl._cache via hooks. Compute the rank
                    # loss BEFORE backpropping, then backward twice on
                    # the retained graph.
                    loss_rank = pacl.rank_loss()
                    has_rank = loss_rank.requires_grad and float(loss_rank.detach()) > 0
                    if has_rank:
                        loss_task.backward(retain_graph=True)
                    else:
                        loss_task.backward()
                    task_grads = {
                        n: p.grad.detach().clone()
                        for n, p in model.named_parameters()
                        if p.grad is not None
                    }
                    if has_rank:
                        optimizer.zero_grad(set_to_none=True)
                        loss_rank.backward()
                        rank_grads = {
                            n: p.grad.detach().clone()
                            for n, p in model.named_parameters()
                            if p.grad is not None
                        }
                        proj = pacl.project_rank_grad(rank_grads)
                    else:
                        proj = {}
                    # Combine and step.
                    for n, p in model.named_parameters():
                        if n not in task_grads and n not in proj:
                            p.grad = None
                            continue
                        g = task_grads.get(n, torch.zeros_like(p))
                        if n in proj:
                            g = g + pacl.cfg.lam * proj[n]
                        p.grad = g
                    optimizer.step()

                if hasattr(base, "post_step"):
                    base.post_step()

                if log_every > 0 and it % log_every == 0:
                    print(
                        f"[task {t + 1}/{T} ep {ep + 1}/{epochs_per_task} "
                        f"it {it}] loss_task={float(loss_task.detach()):.4f}",
                        flush=True,
                    )
        train_loss_traj.append(running_loss / max(1, running_n))

        # End-of-task plasticity diag.
        diag_acc.append(evaluate(model, test_loaders[t], device))

        # End-of-task effective-rank diag on a probe batch.
        if layers_for_probe:
            px, _ = _probe_batch(task, device, n=512)
            model.eval()
            with torch.no_grad():
                _ = model(px)
            if pacl is not None:
                row = {}
                from .pacl import effective_rank as _erank

                for name, H in pacl._cache.items():
                    row[name] = float(_erank(H).detach().cpu())
                pacl._cache.clear()
                rank_traj.append(row)
                rank_floor_traj.append(dict(pacl.rank_floor))
            else:
                row = {}
                for name, H in probe_cache.items():
                    row[name] = float(effective_rank_probe(H))
                probe_cache.clear()
                rank_traj.append(row)

        # Some baselines (EWC) need to update their Fisher / parameter
        # snapshot from the just-finished task; pass a fresh-iterator
        # builder so they can pull batches without consuming the trainer's.
        def _new_iter():
            return _train_iter(task, batch_size)

        try:
            base.end_of_task(train_iter_fn=_new_iter, device=device)
        except TypeError:
            base.end_of_task()
        if pacl is not None:
            pacl.update_rank_floor()
            # Fisher update needs an iterable of (x, y). Use the train iter.
            pacl.update_fisher(_train_iter(task, batch_size), device)
        print(
            f"[task {t + 1} done] diag_acc={diag_acc[-1]:.4f} "
            f"loss={train_loss_traj[-1]:.4f}  "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    # Final-row eval across all T tasks.
    final_row: List[float] = []
    for s in range(T):
        final_row.append(evaluate(model, test_loaders[s], device))

    for h in probe_handles:
        h.remove()

    return TrainerResult(
        diag_acc=diag_acc,
        final_row=final_row,
        rank_traj=rank_traj,
        rank_floor_traj=rank_floor_traj,
        train_loss_traj=train_loss_traj,
        seconds=time.time() - t0,
    )
