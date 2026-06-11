"""Model registry."""
from .mlp import MLP
from .resnet import cifar_resnet18, cifar_resnet32

__all__ = ["MLP", "cifar_resnet18", "cifar_resnet32", "build_model"]


def build_model(name: str, **kw):
    name = name.lower()
    if name == "mlp":
        return MLP(**kw)
    if name in ("cifar_resnet18", "resnet18"):
        return cifar_resnet18(**kw)
    if name in ("cifar_resnet32", "resnet32"):
        return cifar_resnet32(**kw)
    raise ValueError(f"Unknown model: {name}")