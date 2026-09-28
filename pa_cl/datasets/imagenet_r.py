"""Split-ImageNet-R dataset for class-incremental learning.

ImageNet-R (Hendrycks et al., 2021) contains renditions (art, cartoons,
deviantart, graffiti, origami, paintings, sketches, tattoos, toys,
videogame graphics) of 200 ImageNet classes. It tests robustness to
domain shift — the core challenge that motivates PA-CL's plasticity
preservation.

Standard CL setup: 200 classes split into 10 tasks of 20 classes each
(class-incremental). Since ImageNet-R provides no separate train/test
split, we partition each class 80%/20% into train/test (seeded).

All images are resized to 64x64 and pre-loaded onto the target device,
following the same GPU-resident pattern as split_tinyimagenet.py.

Directory layout expected after extraction:
    root/imagenet-r/
      n01443537/*.png
      n01484850/*.png
      ...

Download: https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

IMAGENET_R_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"

try:
    from PIL import Image as _PILImg

    _BILINEAR = _PILImg.Resampling.BILINEAR
except (ImportError, AttributeError):
    _BILINEAR = None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _extract_if_needed(root: str) -> str:
    target = os.path.join(root, "imagenet-r")
    if os.path.isdir(target):
        return target
    tar_path = os.path.join(root, "imagenet-r.tar")
    if not os.path.isfile(tar_path):
        raise FileNotFoundError(
            f"ImageNet-R not found at {target} or {tar_path}. "
            f"Download from {IMAGENET_R_URL} and place in {root}/"
        )
    print("[imagenet-r] extracting tar ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(root)
    if os.path.isdir(os.path.join(root, "imagenet-r")):
        target = os.path.join(root, "imagenet-r")
    return target


def _load_from_hf(img_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load ImageNet-R from HuggingFace dataset axiong/imagenet-r.

    Returns (x, y) where x is (N, 3, img_size, img_size) uint8
    and y is (N,) long with class indices 0..199 (sorted wnid order).
    """
    import numpy as np
    from PIL import Image

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    print("[imagenet-r] loading from Hugging Face (axiong/imagenet-r) ...")
    ds = load_dataset("axiong/imagenet-r", split="test")

    wnids = sorted(set(ds["wnid"]))
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    assert len(wnids) == 200, f"expected 200 classes, got {len(wnids)}"

    images: List[np.ndarray] = []
    labels: List[int] = []
    for item in ds:
        img = item["image"].convert("RGB")
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        images.append(arr)
        labels.append(wnid_to_idx[item["wnid"]])

    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(labels, dtype=torch.long)
    print(f"[imagenet-r] loaded {x.shape[0]} images from HF")
    return x, y


def _load_images(
    data_dir: str,
    img_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load all images from imagenet-r class subdirectories.

    Returns (x, y) where x is (N, 3, img_size, img_size) uint8
    and y is (N,) long with class indices 0..199.
    """
    from PIL import Image
    import numpy as np

    sorted_classes = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    if len(sorted_classes) != 200:
        raise RuntimeError(f"expected 200 class dirs, found {len(sorted_classes)}")

    images: List[np.ndarray] = []
    labels: List[int] = []
    for cls_idx, cls_name in enumerate(sorted_classes):
        cls_dir = os.path.join(data_dir, cls_name)
        files = sorted(
            f
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        )
        for fname in files:
            path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((img_size, img_size), _BILINEAR)
                arr = np.array(img, dtype=np.uint8)
                images.append(arr)
                labels.append(cls_idx)
            except Exception:
                continue

    if not images:
        raise RuntimeError(f"no images loaded from {data_dir}")

    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def _split_train_test(
    x: torch.Tensor,
    y: torch.Tensor,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class 80/20 train/test split."""
    gen = torch.Generator().manual_seed(seed)
    xtr_list, ytr_list, xte_list, yte_list = [], [], [], []
    for c in range(int(y.max().item()) + 1):
        idx = (y == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=gen)]
        n_test = max(1, int(len(perm) * test_ratio))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        xtr_list.append(x[train_idx])
        ytr_list.append(y[train_idx])
        xte_list.append(x[test_idx])
        yte_list.append(y[test_idx])
    xtr = torch.cat(xtr_list)
    ytr = torch.cat(ytr_list)
    xte = torch.cat(xte_list)
    yte = torch.cat(yte_list)
    return xtr, ytr, xte, yte


def _augment_batch(
    x_uint8: torch.Tensor,
    train: bool,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    x = x_uint8.float().div_(255.0)
    if train:
        flip_mask = torch.rand(x.size(0), device=x.device) < 0.5
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
        x_pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
        B = x.size(0)
        sz = x.size(-1)
        oh = torch.randint(0, 9, (B,), device=x.device)
        ow = torch.randint(0, 9, (B,), device=x.device)
        out = torch.empty_like(x)
        for i in range(B):
            out[i] = x_pad[i, :, oh[i] : oh[i] + sz, ow[i] : ow[i] + sz]
        x = out
    return (x - mean) / std


@dataclass
class _ImageNetRSplit:
    x: torch.Tensor
    y: torch.Tensor


class ImageNetRTask:
    """A single class-incremental task: a subset of classes from ImageNet-R."""

    def __init__(
        self,
        train: _ImageNetRSplit,
        test: _ImageNetRSplit,
        class_indices: List[int],
        device: torch.device,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        self.device = device
        self.class_indices = sorted(class_indices)
        self.mean = mean
        self.std = std

        mask_tr = torch.zeros(train.y.shape[0], dtype=torch.bool, device=device)
        for c in class_indices:
            mask_tr |= train.y == c
        self._train_x = train.x[mask_tr]
        self._train_y = train.y[mask_tr]

        mask_te = torch.zeros(test.y.shape[0], dtype=torch.bool, device=device)
        for c in class_indices:
            mask_te |= test.y == c
        self._test_x = test.x[mask_te]
        self._test_y = test.y[mask_te]

    @property
    def n_train(self) -> int:
        return int(self._train_x.shape[0])

    @property
    def n_test(self) -> int:
        return int(self._test_x.shape[0])

    def train_iter(
        self,
        batch_size: int,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
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
            xb_raw = x[s : s + batch_size]
            yb = y[s : s + batch_size]
            xb = _augment_batch(xb_raw, train=True, mean=self.mean, std=self.std)
            yield xb, yb

    def test_iter(
        self, batch_size: int = 256
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self.n_test
        for s in range(0, n, batch_size):
            xb_raw = self._test_x[s : s + batch_size]
            yb = self._test_y[s : s + batch_size]
            xb = _augment_batch(xb_raw, train=False, mean=self.mean, std=self.std)
            yield xb, yb

    def probe_batch(self, n: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        nn_ = min(n, self.n_train)
        xb = _augment_batch(
            self._train_x[:nn_], train=False, mean=self.mean, std=self.std
        )
        return xb, self._train_y[:nn_]


def build_split_imagenet_r(
    root: str = "./data",
    n_tasks: int = 10,
    classes_per_task: int = 20,
    batch_size: int = 64,
    seed: int = 0,
    download: bool = True,
    device: Optional[str] = None,
    img_size: int = 64,
    class_order: Optional[List[int]] = None,
    **_kw,
) -> Tuple[List[ImageNetRTask], List[ImageNetRTask]]:
    """Returns (tasks, tasks) — same list twice to mirror the task API.

    n_tasks * classes_per_task must equal 200.
    """
    if n_tasks * classes_per_task != 200:
        raise ValueError(
            f"n_tasks * classes_per_task must equal 200; got "
            f"{n_tasks}*{classes_per_task} = {n_tasks * classes_per_task}"
        )
    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        data_dir = _extract_if_needed(root)
        print("[imagenet-r] loading images from local directory ...")
        x, y = _load_images(data_dir, img_size=img_size)
    except FileNotFoundError:
        print("[imagenet-r] local data not found, falling back to HuggingFace ...")
        x, y = _load_from_hf(img_size=img_size)
    print(f"[imagenet-r] total: {x.shape}")

    xtr, ytr, xte, yte = _split_train_test(x, y, test_ratio=0.2, seed=seed)
    print(f"[imagenet-r] train: {xtr.shape}, test: {xte.shape}")

    train_split = _ImageNetRSplit(x=xtr.to(dev), y=ytr.to(dev))
    test_split = _ImageNetRSplit(x=xte.to(dev), y=yte.to(dev))
    mean = IMAGENET_MEAN.to(dev)
    std = IMAGENET_STD.to(dev)

    if class_order is None:
        class_order = list(range(200))
    else:
        if sorted(class_order) != list(range(200)):
            raise ValueError("class_order must be a permutation of 0..199")

    tasks: List[ImageNetRTask] = []
    for t in range(n_tasks):
        cls = class_order[t * classes_per_task : (t + 1) * classes_per_task]
        tasks.append(ImageNetRTask(train_split, test_split, cls, dev, mean, std))
    return tasks, tasks


def build_full_imagenet_r(
    root: str = "./data",
    batch_size: int = 64,
    seed: int = 0,
    download: bool = True,
    device: Optional[str] = None,
    **_kw,
) -> Tuple[List[ImageNetRTask], List[ImageNetRTask]]:
    """For Joint upper-bound training: one task containing all 200 classes."""
    return build_split_imagenet_r(
        root=root,
        n_tasks=1,
        classes_per_task=200,
        batch_size=batch_size,
        seed=seed,
        download=download,
        device=device,
    )
