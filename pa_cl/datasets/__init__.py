"""Dataset registry for continual benchmarks."""

from .permuted_mnist import build_permuted_mnist
from .fake_pmnist import build_fake_pmnist
from .split_cifar100 import build_split_cifar100, build_full_cifar100
from .split_tinyimagenet import build_split_tinyimagenet, build_full_tinyimagenet
from .imagenet_r import build_split_imagenet_r, build_full_imagenet_r
from .fake_cifar100 import build_fake_split_cifar100

__all__ = [
    "build_permuted_mnist",
    "build_fake_pmnist",
    "build_split_cifar100",
    "build_full_cifar100",
    "build_split_tinyimagenet",
    "build_full_tinyimagenet",
    "build_split_imagenet_r",
    "build_full_imagenet_r",
    "build_fake_split_cifar100",
    "build_dataset",
]


def build_dataset(name: str, **kw):
    name = name.lower()
    if name == "permuted_mnist":
        return build_permuted_mnist(**kw)
    if name == "fake_pmnist":
        return build_fake_pmnist(**kw)
    if name in ("split_cifar100", "scifar100"):
        return build_split_cifar100(**kw)
    if name in ("full_cifar100", "joint_cifar100"):
        return build_full_cifar100(**kw)
    if name in ("split_tinyimagenet", "stinyimagenet", "tinyimagenet"):
        return build_split_tinyimagenet(**kw)
    if name in ("full_tinyimagenet", "joint_tinyimagenet"):
        return build_full_tinyimagenet(**kw)
    if name in ("split_imagenet_r", "simagenet_r", "imagenet_r"):
        return build_split_imagenet_r(**kw)
    if name in ("full_imagenet_r", "joint_imagenet_r"):
        return build_full_imagenet_r(**kw)
    if name in ("fake_split_cifar100", "fake_scifar100"):
        return build_fake_split_cifar100(**kw)
    raise ValueError(f"Unknown dataset: {name}")
