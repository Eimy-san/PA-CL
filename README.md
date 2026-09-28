# Preserving Representational Plasticity in Deep Continual Learning with One-Sided Effective-Rank Floors

Reference implementation, baselines, and experiment scripts for the paper:

**"Preserving Representational Plasticity in Deep Continual Learning with One-Sided Effective-Rank Floors"**

*Double-anonymous manuscript; author identities withheld during peer
review.*

## Abstract

In sequential learning, the hidden representations of a deep network can
lose their effective rank: the singular spectrum of layer activations
concentrates onto a few dominant directions until the layer no longer
carries the capacity to represent anything new. Spectral collapse offers
a single measurable window onto the two classical failures of continual
learning, catastrophic forgetting and loss of plasticity, and this paper
turns it into a mechanism. PA-CL attaches to any base continual learner
as a plug-in that holds the effective rank of monitored features at a
running floor: a smooth one-sided hinge supplies plasticity pressure
exactly when the representation runs low, and a Fisher-weighted
attenuation of the resulting gradient bounds the interference it can
cause to what has already been learned. Three formal results back the
design: an Eckart-Young capacity-collapse lemma, a
plasticity-preservation theorem with an explicitly characterized
stationary floor under a Polyak-Lojasiewicz condition, and a stability
bound for the attenuated gradient. Empirically, PA-CL preserves 2.06x
higher effective rank than ERM across 200 permuted-MNIST tasks
(+3.12 pp accuracy, +2.82 pp backward transfer), raises the effective
rank of ER, DER++, ER-ACE and X-DER stacks across the Class-IL
benchmarks (11 of 12 stacking combinations), and on domain-shifted
Split-ImageNet-R its combination with ER-ACE attains the best accuracy
(9.13%), the best backward transfer (-24.32), and the highest effective
rank (284.5) simultaneously. Whether rank preservation converts into
accuracy depends on which bottleneck - representational capacity or
output-space forgetting - is active, and the mechanism is built to
compose with replay and output-space correction.

## Repository layout

```
PA-CL/
├── pa_cl/
│   ├── pacl.py              # Effective-rank regularizer + Fisher-weighted attenuation
│   ├── trainer.py           # Continual training loop, metrics
│   ├── metrics.py           # ACC / BWT / plasticity decay / effective rank
│   ├── models/              # ResNet-18, 2-layer MLP
│   │   ├── mlp.py
│   │   └── resnet.py
│   ├── baselines/           # ERM, ER, DER++, EWC, CBP, A-GEM, ER-ACE, CLS-ER, X-DER (reduced)
│   │   ├── erm.py
│   │   ├── er.py
│   │   ├── der.py
│   │   ├── ewc.py
│   │   ├── cbp.py
│   │   ├── agem.py
│   │   ├── er_ace.py
│   │   ├── cls_er.py
│   │   └── xder.py
│   └── datasets/            # SplitCIFAR-100, Split-Tiny-ImageNet, Split-ImageNet-R, CPMNIST
│       ├── split_cifar100.py
│       ├── split_tinyimagenet.py
│       ├── imagenet_r.py
│       ├── permuted_mnist.py
│       ├── fake_cifar100.py
│       └── fake_pmnist.py
├── configs/                 # YAML configs per experiment (all seeds/hyperparameters committed)
├── scripts/                 # CLI entry points + sweep launchers + figure regeneration
├── runbook/                 # Experiment runbook + per-seed summaries + aggregate results
│   ├── RUNBOOK.md           # Full experiment plan and gate criteria
│   └── */_aggregate/        # Summary tables and gate evaluations
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
# 1. Install dependencies (CUDA assumed)
pip install -r requirements.txt

# 2. Smoke test (~3 min on any GPU)
python scripts/run_experiment.py --config configs/smoke_pmnist.yaml --method pacl --seed 0
```

The smoke test verifies that the data pipeline, effective-rank computation, Fisher accumulator, and PA-CL's two-backward path all work correctly.

## Reproducing the paper's results

All seeds, hyperparameters, and dataset splits are committed in `configs/`. See `runbook/RUNBOOK.md` for the full experiment plan.

| Experiment | Command | Approx. GPU-hours |
|---|---|---|
| PoC (CPMNIST-200, 3 methods x 3 seeds) | `bash scripts/run_poc.sh` | ~0.6 h |
| Main table (CIFAR-100, 14 methods x 5 seeds) | `bash scripts/run_cifar100_sweep.sh` | ~18 h |
| Main table (TinyImageNet, 14 methods x 5 seeds) | `bash scripts/run_tinyimagenet_sweep.sh` | ~73 h |
| Main table (ImageNet-R, 14 methods x 5 seeds) | `bash scripts/run_imagenet_r.sh` | ~19 h |
| Modern baselines (ER-ACE / CLS-ER / X-DER + PA-CL stacks) | `bash scripts/run_modern_baselines.sh` | included above |
| λ sensitivity (5 values x 5 seeds) | `bash scripts/run_lambda_sweep.sh` | ~9 h |
| Fisher-attenuation on/off ablation | `bash scripts/run_proj_sweep.sh` | ~1 h |
| Buffer-capacity sweep (5 sizes x 5 seeds) | `bash scripts/run_buffer_sweep.sh` | ~8.5 h |
| Layer-selection ablation (3 configs x 3 seeds) | `bash scripts/run_layer_sweep.sh` | ~4 h |
| Theorem diagnostic (CPU only) | `python scripts/theorem_diagnostic.py` | ~0 h |
| Long-horizon (1000 tasks, 3 methods x 3 seeds) | `bash scripts/run_long_horizon.sh` | ~5 h |

Tested on a single NVIDIA RTX 4090 (24 GB) with PyTorch 2.11.0 / CUDA 12.8. Total compute budget is approximately 140 GPU-hours.

**Note on X-DER:** the X-DER baseline is a *reduced* variant without the SimCLR-based future-preparation consistency term (which requires an augmentation pipeline outside our shared evaluation stack). Both X-DER and PA-CL+X-DER use the same reduced base, so stacking comparisons between the two arms are internally consistent; absolute X-DER numbers are conservative.

## Figures

All figures in the paper are generated programmatically from the committed runbook logs:

```bash
python scripts/regenerate_all_figures.py   # requires the runbook data
python scripts/theorem_diagnostic.py       # CPU-only synthetic diagnostic
```

## Citation

A formal citation entry will be added here upon acceptance. The
manuscript is currently under double-anonymous peer review; author
identities are withheld in this repository during review.

## License

This repository is released for academic research purposes. Please contact the authors for commercial use.
