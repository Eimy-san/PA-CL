"""Split-CIFAR-100 dataset for class-incremental learning.

Standard CL benchmark: 100 classes are split into N_TASKS contiguous
groups (default: 10 tasks of 10 classes each). The classifier is a
single 100-way head shared across tasks; at any inference time the
prediction is the argmax over all 100 logits (no task-id provided).

Like our permuted_mnist builder, we pre-load the entire CIFAR-100 split
onto the target device (~600 MB at float32; cheap on the 4090) and
expose `SplitCIFAR100Task` objects that look identical to
`PermutedTask` (same iter interface, so the trainer is reused unchanged).

We follow the Mammoth convention (Buzzega et al., 2020):
  - Normalize with channel-wise mean/std computed on CIFAR-100 train.
  - Train transform: random crop 32x32 padding=4 + horizontal flip.
  - Test transform: deterministic.
We materialize both *raw* (uint8, on-device) and a per-batch augmentor
that runs on GPU; this keeps the dataloader off the CPU path entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision


CIFAR100_MEAN = torch.tensor([0.5071, 0.4866, 0.4409]).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


def _load_cifar100(root: str, train: bool, download: bool
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.CIFAR100(root=root, train=train, download=download)
    # ds.data is (N, 32, 32, 3) uint8 numpy; convert to (N, 3, 32, 32) uint8 tensor.
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def _augment_batch(x_uint8: torch.Tensor, train: bool,
                   mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Convert a (B, 3, 32, 32) uint8 batch to a normalized float batch.

    Train mode also applies random crop (padding=4) + random horizontal flip,
    both done on-device for speed."""
    x = x_uint8.float().div_(255.0)
    if train:
        # Random horizontal flip per sample.
        flip_mask = (torch.rand(x.size(0), device=x.device) < 0.5)
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
        # Reflect-pad 4 on each side, then random crop back to 32x32.
        x_pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
        B = x.size(0)
        oh = torch.randint(0, 9, (B,), device=x.device)
        ow = torch.randint(0, 9, (B,), device=x.device)
        # Gather the crops via grid_sample-like indexing.
        # Simpler / correct: build an index tensor per sample.
        out = torch.empty_like(x)
        for i in range(B):
            out[i] = x_pad[i, :, oh[i]:oh[i] + 32, ow[i]:ow[i] + 32]
        x = out
    return (x - mean) / std


@dataclass
class _CIFAR100Split:
    x: torch.Tensor   # (N, 3, 32, 32) uint8 on device
    y: torch.Tensor   # (N,) long on device


class SplitCIFAR100Task:
    """A single class-incremental task: a subset of classes from CIFAR-100."""

    def __init__(self, train: _CIFAR100Split, test: _CIFAR100Split,
                 class_indices: List[int], device: torch.device,
                 mean: torch.Tensor, std: torch.Tensor):
        self.device = device
        self.class_indices = sorted(class_indices)
        self.mean = mean
        self.std = std
        # Subset by class.
        mask_tr = torch.zeros(train.y.shape[0], dtype=torch.bool, device=device)
        for c in class_indices:
            mask_tr |= (train.y == c)
        self._train_x = train.x[mask_tr]
        self._train_y = train.y[mask_tr]
        mask_te = torch.zeros(test.y.shape[0], dtype=torch.bool, device=device)
        for c in class_indices:
            mask_te |= (test.y == c)
        self._test_x = test.x[mask_te]
        self._test_y = test.y[mask_te]

    @property
    def n_train(self) -> int:
        return int(self._train_x.shape[0])

    @property
    def n_test(self) -> int:
        return int(self._test_x.shape[0])

    def train_iter(self, batch_size: int, shuffle: bool = True,
                   generator: Optional[torch.Generator] = None
                   ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self.n_train
        if shuffle:
            order = torch.randperm(n, generator=generator, device=self.device)
            x = self._train_x[order]
            y = self._train_y[order]
        else:
            x = self._train_x
            y = self._train_y
        for s in range(0, n, batch_size):
            xb_raw = x[s:s + batch_size]
            yb = y[s:s + batch_size]
            xb = _augment_batch(xb_raw, train=True, mean=self.mean, std=self.std)
            yield xb, yb

    def test_iter(self, batch_size: int = 512
                  ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self.n_test
        for s in range(0, n, batch_size):
            xb_raw = self._test_x[s:s + batch_size]
            yb = self._test_y[s:s + batch_size]
            xb = _augment_batch(xb_raw, train=False, mean=self.mean, std=self.std)
            yield xb, yb

    def probe_batch(self, n: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        nn_ = min(n, self.n_train)
        xb = _augment_batch(self._train_x[:nn_], train=False,
                            mean=self.mean, std=self.std)
        return xb, self._train_y[:nn_]


def build_split_cifar100(
    root: str = "./data",
    n_tasks: int = 10,
    classes_per_task: int = 10,
    batch_size: int = 64,
    seed: int = 0,
    download: bool = True,
    device: Optional[str] = None,
    class_order: Optional[List[int]] = None,
    **_kw,
) -> Tuple[List[SplitCIFAR100Task], List[SplitCIFAR100Task]]:
    """Returns (tasks, tasks) -- same list twice to mirror PermutedTask API.

    `class_order`: optional 100-element permutation of [0,99]. If not
    provided, classes are partitioned in sorted order (Mammoth default for
    Split-CIFAR-100); pass a seed-derived permutation for the "random class
    order" variant (often used in CL benchmarks for robustness).
    """
    if n_tasks * classes_per_task != 100:
        raise ValueError(
            f"n_tasks * classes_per_task must equal 100; got "
            f"{n_tasks}*{classes_per_task} = {n_tasks * classes_per_task}"
        )
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    xtr, ytr = _load_cifar100(root, train=True, download=download)
    xte, yte = _load_cifar100(root, train=False, download=download)
    train_split = _CIFAR100Split(x=xtr.to(dev), y=ytr.to(dev))
    test_split = _CIFAR100Split(x=xte.to(dev), y=yte.to(dev))
    mean = CIFAR100_MEAN.to(dev)
    std = CIFAR100_STD.to(dev)

    if class_order is None:
        class_order = list(range(100))
    else:
        if sorted(class_order) != list(range(100)):
            raise ValueError("class_order must be a permutation of 0..99")

    tasks: List[SplitCIFAR100Task] = []
    for t in range(n_tasks):
        cls = class_order[t * classes_per_task:(t + 1) * classes_per_task]
        tasks.append(SplitCIFAR100Task(train_split, test_split, cls, dev, mean, std))
    return tasks, tasks


def build_full_cifar100(
    root: str = "./data",
    batch_size: int = 64,
    seed: int = 0,
    download: bool = True,
    device: Optional[str] = None,
    **_kw,
) -> Tuple[List[SplitCIFAR100Task], List[SplitCIFAR100Task]]:
    """For Joint upper-bound training: one 'task' containing all 100 classes."""
    return build_split_cifar100(
        root=root, n_tasks=1, classes_per_task=100,
        batch_size=batch_size, seed=seed, download=download, device=device,
    )