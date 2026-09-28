"""Split-Tiny-ImageNet dataset for class-incremental learning.

Tiny-ImageNet is a 200-class, 64x64 downsized subset of ImageNet with
500 training images and 50 validation images per class. We treat the
validation split as the test set (standard CL convention).

Standard CL benchmark: 200 classes split into N_TASKS contiguous groups
(default: 10 tasks of 20 classes each). The classifier is a single
200-way head shared across tasks; at inference, prediction is the argmax
over all 200 logits (no task-id provided).

Follows the same on-device, GPU-resident pattern as split_cifar100.py:
all images are pre-loaded onto the target device and augmented on-GPU.

Directory layout expected (standard Tiny-ImageNet download):
    root/
      train/
        n01443537/images.n01443537_*.JPEG
        ...
      val/
        images.txt
        val_annotations.txt

If the data directory is not found, we download from the Stanford CS231n
mirror and extract automatically.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

# Channel-wise statistics computed on the Tiny-ImageNet train split.
TINYIMAGENET_MEAN = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
TINYIMAGENET_STD = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)


def _download_and_extract(root: str) -> str:
    """Download Tiny-ImageNet to root/tiny-imagenet-200/ if not present."""
    target = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(target) and os.path.isdir(os.path.join(target, "train")):
        return target
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    if not os.path.isfile(zip_path):
        print(f"[tiny-imagenet] downloading from {TINYIMAGENET_URL} ...")
        urllib.request.urlretrieve(TINYIMAGENET_URL, zip_path)
    print("[tiny-imagenet] extracting ...")
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    if os.path.isdir(os.path.join(root, "tiny-imagenet-200")):
        target = os.path.join(root, "tiny-imagenet-200")
    _restructure_val(target)
    return target


def _restructure_val(tiny_root: str) -> None:
    """Restructure val/ from flat files to per-class subdirectories."""
    val_dir = os.path.join(tiny_root, "val")
    ann_path = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.isfile(ann_path):
        return
    images_dir = os.path.join(val_dir, "images")
    if not os.path.isdir(images_dir):
        return
    with open(ann_path) as f:
        annotations = {}
        for line in f:
            parts = line.strip().split("\t")
            annotations[parts[0]] = parts[1]
    for img_name, cls in annotations.items():
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(cls_dir, img_name)
        if os.path.isfile(src):
            shutil.move(src, dst)
    shutil.rmtree(images_dir, ignore_errors=True)


def _load_images_from_dir(
    class_dirs_root: str,
    class_to_idx: dict,
    max_per_class: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load all images from a directory of class subdirectories.

    Returns (x, y) where x is (N, 3, 64, 64) uint8 and y is (N,) long.
    Uses PIL for JPEG decoding.
    """
    from PIL import Image
    import numpy as np

    images: List[np.ndarray] = []
    labels: List[int] = []
    sorted_classes = sorted(class_to_idx.keys())
    for cls_name in sorted_classes:
        cls_idx = class_to_idx[cls_name]
        cls_dir = os.path.join(class_dirs_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(
            f
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        )
        if max_per_class is not None:
            files = files[:max_per_class]
        for fname in files:
            path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
                arr = np.array(img, dtype=np.uint8)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                images.append(arr)
                labels.append(cls_idx)
            except Exception:
                continue
    if not images:
        raise RuntimeError(f"no images loaded from {class_dirs_root}")
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def _load_tinyimagenet_hf(
    root: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Tiny-ImageNet from Hugging Face Hub (hf-mirror.com).

    Returns (xtr, ytr, xte, yte) with x as (N, 3, 64, 64) uint8 tensors.
    """
    import numpy as np

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    print("[tiny-imagenet] loading from Hugging Face Hub (hf-mirror.com) ...")
    ds = load_dataset("Maysee/tiny-imagenet")

    def _convert(split):
        n = len(ds[split])
        imgs = np.zeros((n, 64, 64, 3), dtype=np.uint8)
        labels = np.zeros(n, dtype=np.int64)
        for i, item in enumerate(ds[split]):
            img = np.array(item["image"].convert("RGB"), dtype=np.uint8)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            imgs[i] = img
            labels[i] = item["label"]
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
        y = torch.from_numpy(labels)
        return x, y

    xtr, ytr = _convert("train")
    print(f"[tiny-imagenet] train: {xtr.shape}")
    xte, yte = _convert("valid")
    print(f"[tiny-imagenet] val: {xte.shape}")
    return xtr, ytr, xte, yte


def _augment_batch(
    x_uint8: torch.Tensor,
    train: bool,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Convert (B, 3, 64, 64) uint8 to normalized float; augment if train."""
    x = x_uint8.float().div_(255.0)
    if train:
        flip_mask = torch.rand(x.size(0), device=x.device) < 0.5
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
        x_pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
        B = x.size(0)
        oh = torch.randint(0, 9, (B,), device=x.device)
        ow = torch.randint(0, 9, (B,), device=x.device)
        out = torch.empty_like(x)
        for i in range(B):
            out[i] = x_pad[i, :, oh[i] : oh[i] + 64, ow[i] : ow[i] + 64]
        x = out
    return (x - mean) / std


@dataclass
class _TinyImageNetSplit:
    x: torch.Tensor
    y: torch.Tensor


class SplitTinyImageNetTask:
    """A single class-incremental task: a subset of classes from Tiny-ImageNet."""

    def __init__(
        self,
        train: _TinyImageNetSplit,
        test: _TinyImageNetSplit,
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


def build_split_tinyimagenet(
    root: str = "./data",
    n_tasks: int = 10,
    classes_per_task: int = 20,
    batch_size: int = 64,
    seed: int = 0,
    download: bool = True,
    device: Optional[str] = None,
    class_order: Optional[List[int]] = None,
    **_kw,
) -> Tuple[List[SplitTinyImageNetTask], List[SplitTinyImageNetTask]]:
    """Returns (tasks, tasks) -- same list twice to mirror the PermutedTask API.

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

    data_dir = os.path.join(root, "tiny-imagenet-200")
    use_hf = False
    if os.path.isdir(os.path.join(data_dir, "train")) and download:
        pass
    elif download:
        try:
            xtr_hf, ytr_hf, xte_hf, yte_hf = _load_tinyimagenet_hf(root)
            use_hf = True
        except Exception as e:
            print(f"[tiny-imagenet] HF loading failed ({e}), trying Stanford URL ...")
            data_dir = _download_and_extract(root)

    if use_hf:
        xtr, ytr = xtr_hf, ytr_hf
        xte, yte = xte_hf, yte_hf
    else:
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")

        sorted_wordnet_ids = sorted(
            d
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )
        if len(sorted_wordnet_ids) != 200:
            raise RuntimeError(
                f"expected 200 class dirs, found {len(sorted_wordnet_ids)}"
            )
        class_to_idx = {wnid: i for i, wnid in enumerate(sorted_wordnet_ids)}

        print("[tiny-imagenet] loading training images ...")
        xtr, ytr = _load_images_from_dir(train_dir, class_to_idx)
        print(f"[tiny-imagenet] train: {xtr.shape}")

        print("[tiny-imagenet] loading validation images ...")
        xte, yte = _load_images_from_dir(val_dir, class_to_idx)
        print(f"[tiny-imagenet] val: {xte.shape}")

    train_split = _TinyImageNetSplit(x=xtr.to(dev), y=ytr.to(dev))
    test_split = _TinyImageNetSplit(x=xte.to(dev), y=yte.to(dev))
    mean = TINYIMAGENET_MEAN.to(dev)
    std = TINYIMAGENET_STD.to(dev)

    if class_order is None:
        class_order = list(range(200))
    else:
        if sorted(class_order) != list(range(200)):
            raise ValueError("class_order must be a permutation of 0..199")

    tasks: List[SplitTinyImageNetTask] = []
    for t in range(n_tasks):
        cls = class_order[t * classes_per_task : (t + 1) * classes_per_task]
        tasks.append(
            SplitTinyImageNetTask(train_split, test_split, cls, dev, mean, std)
        )
    return tasks, tasks


def build_full_tinyimagenet(
    root: str = "./data",
    batch_size: int = 64,
    seed: int = 0,
    download: bool = True,
    device: Optional[str] = None,
    **_kw,
) -> Tuple[List[SplitTinyImageNetTask], List[SplitTinyImageNetTask]]:
    """For Joint upper-bound training: one 'task' containing all 200 classes."""
    return build_split_tinyimagenet(
        root=root,
        n_tasks=1,
        classes_per_task=200,
        batch_size=batch_size,
        seed=seed,
        download=download,
        device=device,
    )
