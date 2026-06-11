"""Continual Backprop (CBP) baseline.

Reference: Dohare et al., "Loss of plasticity in deep continual learning",
Nature 632 (2024). The mechanism: at every step, a small fraction of
units in each hidden layer is reinitialized. The units chosen are the
lowest-utility units that have passed a maturity threshold.

Utility (mean over recent mini-batches of):
    u_i = | mean activation of unit i | * sum_j | outgoing weight w_{j,i} |

After reinitialization the unit's incoming weights are drawn from the
initialization distribution and the corresponding outgoing weights are
zeroed (so the rest of the network is not perturbed at the moment of
replacement). The unit's age counter is reset to 0.

Hyperparameters follow the Nature paper's defaults for MLPs:
    replacement_rate = 1e-4     (fraction of units replaced per step)
    decay_rate       = 0.99     (utility EMA decay)
    maturity_thresh  = 100      (steps before a unit is eligible for replacement)

This implementation targets a `pa_cl.models.mlp.MLP` whose hidden
layers live under `model.hidden` as `nn.Linear` modules and whose
output head is `model.head`. It is straightforward to extend to convs.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBP:
    """Continual Backprop wrapper for an MLP.

    Use as a base learner: provides .task_loss() like ERM, and .end_of_task()
    is a no-op. The unit-reset step is performed inside .post_step() which
    the trainer should call after optimizer.step() when the baseline is CBP.
    """

    def __init__(
        self,
        model: nn.Module,
        replacement_rate: float = 1e-4,
        decay_rate: float = 0.99,
        maturity_threshold: int = 100,
        verbose: bool = False,
    ):
        self.model = model
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.verbose = verbose
        # Per-unit running utility (EMA) and age, per hidden layer.
        self._util: Dict[str, torch.Tensor] = {}
        self._age: Dict[str, torch.Tensor] = {}
        # Pre-activation/activation cache filled by forward hooks.
        self._act_cache: Dict[str, torch.Tensor] = {}
        # Per-layer fractional-replacement accumulator (we replace a whole
        # unit when this exceeds 1.0; standard CBP trick).
        self._frac_accum: Dict[str, float] = {}
        # Map: hidden layer name -> Linear module that *consumes* its
        # activations (the "outgoing" weight). For our MLP the order is
        # h1 -> h2 -> head, so outgoing(h1)=h2, outgoing(h2)=head.
        self._outgoing: Dict[str, nn.Linear] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _register(self) -> None:
        if not hasattr(self.model, "hidden") or not hasattr(self.model, "head"):
            raise AttributeError(
                "CBP wrapper currently expects model.hidden (ModuleDict of "
                "nn.Linear) and model.head (nn.Linear); generalize to "
                "support other architectures as needed."
            )
        hidden_names = list(self.model.hidden.keys())
        layers = [self.model.hidden[n] for n in hidden_names] + [self.model.head]
        for i, name in enumerate(hidden_names):
            layer: nn.Linear = self.model.hidden[name]
            out_features = layer.out_features
            self._util[name] = torch.zeros(out_features, device=layer.weight.device)
            self._age[name] = torch.zeros(out_features, device=layer.weight.device,
                                          dtype=torch.long)
            self._frac_accum[name] = 0.0
            self._outgoing[name] = layers[i + 1]
            self._handles.append(
                self.model.hidden[name].register_forward_hook(
                    self._make_hook(name)
                )
            )

    def _make_hook(self, name: str):
        def _hook(_module, _inp, out):
            # Cache post-Linear (pre-activation) for utility computation;
            # for ReLU MLPs Dohare et al. measure utility on the
            # post-activation, which we approximate with relu(out).
            self._act_cache[name] = out
        return _hook

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return F.cross_entropy(logits, y)

    # ------------------------------------------------------------------
    # Per-step utility update + selective reinitialization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def post_step(self) -> None:
        """Update utility EMA, age, then reinitialize lowest-utility units."""
        for name, act in self._act_cache.items():
            outgoing: nn.Linear = self._outgoing[name]
            # Activation magnitude: mean |act_i| over the batch (post-ReLU).
            a = F.relu(act).abs().mean(dim=0)  # (out_features,)
            # Outgoing weight magnitude: sum_j |w_{j,i}|.
            w_out = outgoing.weight.abs().sum(dim=0)  # (in_features,) == (this layer's out_features,)
            instant = a * w_out
            u = self._util[name]
            self._util[name] = self.decay_rate * u + (1 - self.decay_rate) * instant
            self._age[name] = self._age[name] + 1

            # Fractional accumulator: each step we *want* to replace
            # replacement_rate * n_units units.
            n_units = u.shape[0]
            self._frac_accum[name] += self.replacement_rate * n_units
            n_to_replace = int(self._frac_accum[name])
            if n_to_replace <= 0:
                continue
            self._frac_accum[name] -= n_to_replace

            eligible = self._age[name] >= self.maturity_threshold
            if eligible.sum().item() == 0:
                continue

            util_eligible = self._util[name].clone()
            util_eligible[~eligible] = float("inf")
            n_to_replace = min(n_to_replace, int(eligible.sum().item()))
            _, idx = torch.topk(util_eligible, n_to_replace, largest=False)
            self._reinit_units(name, idx)

        self._act_cache.clear()

    @torch.no_grad()
    def _reinit_units(self, name: str, idx: torch.Tensor) -> None:
        layer: nn.Linear = self.model.hidden[name]
        outgoing: nn.Linear = self._outgoing[name]
        # Reinitialize the incoming weights and bias of these units.
        # Match nn.Linear default init (Kaiming uniform with bound 1/sqrt(in)).
        bound = 1.0 / (layer.in_features ** 0.5)
        new_w = torch.empty(len(idx), layer.in_features,
                            device=layer.weight.device).uniform_(-bound, bound)
        layer.weight.data[idx] = new_w
        if layer.bias is not None:
            layer.bias.data[idx] = 0.0
        # Zero the outgoing weights: the reset unit must not perturb the
        # downstream computation at the moment of replacement.
        outgoing.weight.data[:, idx] = 0.0
        # Reset utility and age.
        self._util[name][idx] = 0.0
        self._age[name][idx] = 0
        if self.verbose:
            print(f"[CBP] layer {name}: reset {len(idx)} units")

    # ------------------------------------------------------------------
    # End-of-task / cleanup
    # ------------------------------------------------------------------
    def end_of_task(self, *_args, **_kw) -> None:
        return None

    def detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()