# PA-CL: Plasticity-Aware Continual Learning via Effective-Rank Regularization

Reference implementation, baselines, and experiment scripts for the paper:

**"PA-CL: Plasticity-Aware Continual Learning via Effective-Rank Regularization"**

*Guoping You, Yudan Hu, Jingze Li*

Submitted to *Expert Systems with Applications* (Elsevier).

## Abstract

Continual-learning agents that optimize solely for stability often suffer progressive loss of representational plasticity. We propose **PA-CL**, a lightweight plug-in that (1) enforces a one-sided effective-rank floor on hidden activations and (2) projects the rank gradient onto the Fisher-null subspace so that stability is preserved without explicit task boundaries. On Split-CIFAR-100, PA-CL improves average accuracy by +3.12 pp over the ER baseline and maintains 2.06× higher effective rank than ERM on Continual-Permuted-MNIST-200.

## Repository layout

```
PA-CL/
├── pa_cl/
│   ├── pacl.py              # Effective-rank regularizer + Fisher-null projection
│   ├── trainer.py           # Continual training loop, metrics
│   ├── metrics.py           # ACC / BWT / FWT / plasticity decay / rank
│   ├── models/              # ResNet-18, 2-layer MLP
│   │   ├── mlp.py
│   │   └── resnet.py
│   ├── baselines/           # ERM, ER, DER++, EWC, CBP, ...
│   │   ├── erm.py
│   │   ├── er.py
│   │   ├── der.py
│   │   ├── ewc.py
│   │   ├── cbp.py
│   │   └── agem.py
│   └── datasets/            # SplitCIFAR-100, CPMNIST, ...
│       ├── split_cifar100.py
│       ├── permuted_mnist.py
│       ├── fake_cifar100.py
│       └── fake_pmnist.py
├── configs/                 # YAML configs per experiment
├── scripts/                 # CLI entry points + sweep launchers
├── runbook/                 # Experiment runbook + aggregate results
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
| PoC (CPMNIST-200, 3 methods) | `bash scripts/run_poc.sh` | ~2 h |
| Main table (CIFAR-100 sweep) | `bash scripts/run_cifar100_sweep.sh` | ~80 h |
| λ sensitivity | `bash scripts/run_lambda_sweep.sh` | ~20 h |
| Projection ablation | `bash scripts/run_proj_sweep.sh` | ~15 h |
| Long-horizon (1000 tasks) | `bash scripts/run_long_horizon.sh` | ~10 h |

Tested on a single NVIDIA A100-40GB with PyTorch 2.5.1 / Python 3.12 / CUDA 12.4.

## Citation

If you find this work useful, please cite:

```bibtex
@article{you2026pacl,
  title   = {{PA-CL}: Plasticity-Aware Continual Learning via Effective-Rank Regularization},
  author  = {You, Guoping and Hu, Yudan and Li, Jingze},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review}
}
```

## License

This repository is released for academic research purposes. Please contact the authors for commercial use.