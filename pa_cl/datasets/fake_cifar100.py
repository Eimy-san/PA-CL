"""Synthetic CIFAR-100-shaped dataset for offline pipeline tests.

Generates random uint8 (B, 3, 32, 32) images and synthetic labels in
[0, 99] derived from per-channel sums (so the labels are determined by
the input and learnable). Intended ONLY for engineering smoke tests in
sandboxed environments. Do not use it for any scientific claim.

Uses the same SplitCIFAR100Task interface as the real builder so the
rest of the pipeline (trainer, baselines, PA-CL) is exercised
end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch

from .split_cifar100 import (SplitCIFAR100Task, _CIFAR100Split,
                             CIFAR100_MEAN, CIFAR100_STD)


def build_fake_split_cifar100(
    n_tasks: int = 5,
    classes_per_task: int = 4,
    batch_size: int = 32,
    seed: int = 0,
    n_train_per_class: int = 50,
    n_test_per_class: int = 20,
    device: Optional[str] = None,
    **_kw,
) -> Tuple[List[SplitCIFAR100Task], List[SplitCIFAR100Task]]:
    """Tiny synthetic CIFAR-100 stand-in. Total classes = n_tasks * classes_per_task.

    Default values yield a 5*4 = 20 class problem with ~ 50*20=1000 train,
    20*20=400 test samples; runs end-to-end in seconds on CPU.
    """
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    n_classes = n_tasks * classes_per_task
    g = torch.Generator().manual_seed(seed)
    # Make labels deterministic from input: assign per-class means.
    centers = (torch.rand((n_classes, 3, 32, 32), generator=g) * 255).to(torch.uint8)
    train_x_list, train_y_list = [], []
    test_x_list, test_y_list = [], []
    for c in range(n_classes):
        # Per-class samples = center + small uniform noise.
        for n_, dst_x, dst_y in ((n_train_per_class, train_x_list, train_y_list),
                                 (n_test_per_class, test_x_list, test_y_list)):
            noise = (torch.rand((n_, 3, 32, 32), generator=g) * 30 - 15).to(torch.int16)
            samples = (centers[c].to(torch.int16) + noise).clamp(0, 255).to(torch.uint8)
            dst_x.append(samples)
            dst_y.append(torch.full((n_,), c, dtype=torch.long))
    xtr = torch.cat(train_x_list, dim=0).to(dev)
    ytr = torch.cat(train_y_list, dim=0).to(dev)
    xte = torch.cat(test_x_list, dim=0).to(dev)
    yte = torch.cat(test_y_list, dim=0).to(dev)

    train_split = _CIFAR100Split(x=xtr, y=ytr)
    test_split = _CIFAR100Split(x=xte, y=yte)
    mean = CIFAR100_MEAN.to(dev)
    std = CIFAR100_STD.to(dev)

    tasks: List[SplitCIFAR100Task] = []
    for t in range(n_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        tasks.append(SplitCIFAR100Task(train_split, test_split, cls, dev, mean, std))
    return tasks, tasks