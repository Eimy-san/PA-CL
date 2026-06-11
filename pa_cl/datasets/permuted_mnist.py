"""Continual-Permuted-MNIST builder (GPU-resident, fast, memory-lazy).

Each task is MNIST with a fixed permutation of the 784 input pixels.
This is the canonical loss-of-plasticity benchmark from Dohare et al.
(2024, Nature) and Elsayed & Mahmood (2024, arXiv:2404.00781).

The builder returns simple `PermutedTask` objects which expose:

    .train_iter(batch_size, shuffle=True) -> iterator of (x, y)
    .test_iter(batch_size=2048)           -> iterator of (x, y)
    .perm                                 -- (784,) long on device
    .n_train, .n_test                     -- sample counts

Memory model. We keep the two MNIST split tensors on the device exactly
once (train: 60000*784*4 = 188 MB, test: 10000*784*4 = 31 MB) and store
only per-task metadata (`perm`: 784*8 bytes, `indices`: ~40 KB if
samples_per_task=5000) per task. Permutation and index-select happen
lazily inside the iterators on each batch. This keeps total memory
under 220 MB regardless of the number of tasks, which is what
makes 1000-task runs feasible on a 24 GB GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
import torchvision


def _flat_mnist(root: str, train: bool, download: bool
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    base = torchvision.datasets.MNIST(root=root, train=train, download=download)
    x = base.data.float().div_(255.0).view(len(base), -1).contiguous()  # (N, 784)
    y = base.targets.long()
    return x, y


@dataclass
class _MNISTSplit:
    x: torch.Tensor   # (N, 784) on device
    y: torch.Tensor   # (N,) long on device


class PermutedTask:
    """A single permuted-MNIST task. Lazy: stores only `perm` and `indices`."""

    def __init__(self, train_split: _MNISTSplit, test_split: _MNISTSplit,
                 perm: torch.Tensor, train_indices: Optional[torch.Tensor],
                 device: torch.device):
        self.device = device
        self.perm = perm.to(device)
        self._train = train_split
        self._test = test_split
        if train_indices is not None:
            self._train_idx = train_indices.to(device)
        else:
            self._train_idx = None

    @property
    def n_train(self) -> int:
        return (int(self._train_idx.shape[0]) if self._train_idx is not None
                else int(self._train.x.shape[0]))

    @property
    def n_test(self) -> int:
        return int(self._test.x.shape[0])

    # ----- iterators apply the per-task permutation on demand per batch -----
    def _train_indices(self) -> torch.Tensor:
        if self._train_idx is not None:
            return self._train_idx
        return torch.arange(self._train.x.shape[0], device=self.device)

    def train_iter(self, batch_size: int, shuffle: bool = True,
                   generator: Optional[torch.Generator] = None
                   ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        base_idx = self._train_indices()
        n = int(base_idx.shape[0])
        if shuffle:
            order = torch.randperm(n, generator=generator, device=self.device)
            idx = base_idx[order]
        else:
            idx = base_idx
        for s in range(0, n, batch_size):
            ib = idx[s:s + batch_size]
            xb = self._train.x.index_select(0, ib)[:, self.perm]
            yb = self._train.y.index_select(0, ib)
            yield xb, yb

    def test_iter(self, batch_size: int = 2048
                  ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self.n_test
        for s in range(0, n, batch_size):
            xb = self._test.x[s:s + batch_size][:, self.perm]
            yb = self._test.y[s:s + batch_size]
            yield xb, yb

    def probe_batch(self, n: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self._train_indices()
        n = min(n, int(base_idx.shape[0]))
        ib = base_idx[:n]
        xb = self._train.x.index_select(0, ib)[:, self.perm]
        yb = self._train.y.index_select(0, ib)
        return xb, yb


def build_permuted_mnist(
    root: str = "./data",
    n_tasks: int = 5,
    batch_size: int = 128,
    seed: int = 0,
    download: bool = True,
    samples_per_task: Optional[int] = None,
    device: Optional[str] = None,
    **_kw,
) -> Tuple[List[PermutedTask], List[PermutedTask]]:
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    xtr, ytr = _flat_mnist(root, train=True, download=download)
    xte, yte = _flat_mnist(root, train=False, download=download)
    train_split = _MNISTSplit(x=xtr.to(dev), y=ytr.to(dev))
    test_split = _MNISTSplit(x=xte.to(dev), y=yte.to(dev))

    g = torch.Generator().manual_seed(seed)
    perms = [torch.randperm(28 * 28, generator=g) for _ in range(n_tasks)]

    n_train_full = train_split.x.shape[0]
    tasks: List[PermutedTask] = []
    for t, p in enumerate(perms):
        if samples_per_task is not None:
            start = (t * samples_per_task) % n_train_full
            idx_list: List[int] = []
            need = samples_per_task
            cur = start
            while need > 0:
                take = min(need, n_train_full - cur)
                idx_list.extend(range(cur, cur + take))
                need -= take
                cur = 0
            indices = torch.tensor(idx_list, dtype=torch.long)
        else:
            indices = None
        tasks.append(PermutedTask(train_split, test_split, p, indices, dev))

    return tasks, tasks