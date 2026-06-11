"""Synthetic Permuted-MNIST-shaped dataset for offline pipeline tests.

Generates random images of the MNIST shape and a label rule that is
permutation-sensitive (so different permutations correspond to genuinely
different tasks). This is intended ONLY for engineering smoke tests in
sandboxed environments where torchvision cannot download MNIST. Do not
use it for any scientific claim.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class FakeMNIST(Dataset):
    def __init__(self, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand((n, 1, 28, 28), generator=g)
        # Label rule: argmax of sum over rows of pixel slabs (so a
        # permutation of pixels yields a different label distribution).
        bands = self.x.view(n, -1)
        # 10-way split into chunks; label = argmax chunk sum. 784 is
        # not divisible by 10, so use the first 780 features.
        chunks = bands[:, :780].view(n, 10, -1).sum(dim=2)
        self.y = chunks.argmax(dim=1).long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class FakePermutedMNISTTask(Dataset):
    def __init__(self, base: FakeMNIST, perm: torch.Tensor):
        self.base = base
        self.perm = perm

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        x_perm = x.flatten()[self.perm].reshape_as(x)
        # Re-derive label under the permutation so label rule is
        # permutation-sensitive (each task has a different mapping).
        bands = x_perm.view(-1)[:780]
        chunks = bands.view(10, -1).sum(dim=1)
        y = int(chunks.argmax())
        return x_perm, y


def build_fake_pmnist(
    n_tasks: int = 4,
    batch_size: int = 128,
    seed: int = 0,
    n_train: int = 2000,
    n_test: int = 500,
    **_kw,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    train_base = FakeMNIST(n_train, seed=seed)
    test_base = FakeMNIST(n_test, seed=seed + 1000)
    g = torch.Generator().manual_seed(seed)
    perms = [torch.randperm(28 * 28, generator=g) for _ in range(n_tasks)]
    train_loaders, test_loaders = [], []
    for p in perms:
        tr = FakePermutedMNISTTask(train_base, p)
        te = FakePermutedMNISTTask(test_base, p)
        train_loaders.append(
            DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
        )
        test_loaders.append(
            DataLoader(te, batch_size=256, shuffle=False, num_workers=0)
        )
    return train_loaders, test_loaders