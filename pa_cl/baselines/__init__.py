"""Baseline continual learners."""
import inspect

from .erm import ERM
from .cbp import CBP
from .er import ER
from .der import DERpp
from .ewc import EWC
from .agem import AGEM

__all__ = ["ERM", "CBP", "ER", "DERpp", "EWC", "AGEM", "build_baseline"]


_BASELINES = {
    "erm": ERM,
    "cbp": CBP,
    "er": ER,
    "der": DERpp, "derpp": DERpp, "der++": DERpp,
    "ewc": EWC,
    "agem": AGEM,
}


def build_baseline(name: str, model, **kw):
    """Build a baseline by name. Silently ignores kwargs not accepted by
    that baseline's __init__ (so a single shared baseline_kwargs YAML
    block can carry args for the union of all baselines)."""
    name = name.lower()
    if name not in _BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    cls = _BASELINES[name]
    sig = inspect.signature(cls.__init__)
    accepted = {k for k in sig.parameters.keys() if k not in ("self",)}
    filtered = {k: v for k, v in kw.items() if k in accepted}
    return cls(model, **filtered)