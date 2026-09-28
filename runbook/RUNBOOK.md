# PA-CL Experiment Runbook

This runbook is the **single source of truth** for the experiment plan
behind the paper. All experiments listed here are **complete**; their
aggregated results live in `runbook/<exp>/_aggregate/` and the raw
per-seed logs are git-ignored (regenerable from `configs/` + `scripts/`).

## Pipeline overview

```
configs/*.yaml --> scripts/run_experiment.py --> runbook/<exp>/<method>/seed*/
                                                   summary.json, diag_acc.npy,
                                                   final_row.npy, rank_traj.json,
                                                   rank_floor_traj.json (PA-CL),
                                                   train_loss.npy, config.yaml
runbook/<exp>/<method>/seed*/  -->  scripts/aggregate_poc.py  -->
                                                   runbook/<exp>/_aggregate/
                                                     table.md, table.tex,
                                                     gate.{json,txt}, raw.json
```

## Phase 0 — Engineering smoke test (offline, ~30 s on CPU)

Verifies the entire pipeline (data, model, all methods, CBP post-step,
PA-CL two-backward path, Fisher accumulator, rank diagnostics, aggregator,
gate evaluation) without needing a GPU or internet access.

```bash
cd code
pip install -r requirements.txt
for M in erm cbp pacl; do
  python scripts/run_experiment.py \
      --config configs/dryrun_fake.yaml --method $M --seed 0
done
python scripts/aggregate_poc.py --out_dir runbook/dryrun_fake \
       --methods erm cbp pacl
```

## Phase 1 — Consolidated PoC (GPU, ~2 hours on a single A100)

Continual-Permuted-MNIST with 200 tasks for 3 methods (ERM, CBP, PA-CL)
on 3 seeds = 9 cells. Produces the paper's headline plasticity-decay
figure, the effective-rank trajectory figure, and the ACC/BWT/
plasticity-drop table.

**Gate result: PASSED.** End-of-stream penultimate-layer effective rank
57.08 (PA-CL) vs 27.76 (ERM) = 2.06x; ACC +3.12 pp (p = 0.042); BWT
+2.82 pp (p = 0.045). These are the numbers reported in the paper.

## Phase B — Main experiments (all complete, ~140 GPU-hours total on one RTX 4090)

| # | Experiment | Config | Cells | Wall time |
|---|---|---|---|---|
| a | CIFAR-100 main (14 methods x 5 seeds) | `cifar100_main.yaml` | 70 | ~18 h |
| b | Lambda sweep (5 lambda x 5 seeds) | `cifar100_lambda.yaml` | 25 | ~9 h |
| c | Buffer sweep (5 sizes x 5 seeds) | `cifar100_buffer.yaml` | 25 | ~8.5 h |
| d | Layer sweep (3 configs x 3 seeds) | `cifar100_layers.yaml` | 9 | ~4 h |
| e | Tiny-ImageNet main (14 methods x 5 seeds) | `tinyimagenet_main.yaml` | 70 | ~73 h |
| f | ImageNet-R main (14 methods x 5 seeds, 30 epochs/task) | `imagenet_r_main.yaml` | 70 | ~19 h |

The 14 methods: ERM, EWC, A-GEM, ER, DER++, ER-ACE, CLS-ER, X-DER
(reduced; see the README note), PA-CL, and the PA-CL stacks on ER,
DER++, ER-ACE, CLS-ER, X-DER. Class-IL benchmarks use ResNet-18,
20 epochs/task (30 for ImageNet-R), 500-sample reservoir buffer for
replay methods.

One-command execution:

```bash
cd code
bash scripts/run_phase_b.sh          # resumable; skips finished cells
```

## Supporting experiments

- **Theorem diagnostic** (CPU only): `python scripts/theorem_diagnostic.py`
  validates Theorem 1 / Theorem 3 bounds and the Remark 4 proxy claim
  (Pearson r = 0.96).
- **Fisher-attenuation on/off ablation**: `bash scripts/run_proj_sweep.sh`
  (2 arms x 3 seeds on CIFAR-100) plus a 1,000-task variant.
- **Long-horizon stress**: `bash scripts/run_long_horizon.sh`
  (1,000-task CPMNIST, 3 methods x 3 seeds, ~5 h).

## Statistical reporting

All pairwise comparisons in the paper carry paired two-sided t-tests
(`scripts/collect_results.py` regenerates the table from the per-seed
summaries); Wilcoxon signed-rank checks and the n = 5 power caveat are
reported alongside. Non-significant p-values are reported as such.

## Logs

`dryrun_fake/`, `dryrun_cifar/`    — engineering smoke runs.
`poc_cpmnist200/`                  — Phase 1 PoC.
`poc_cpmnist1000/`, `poc_cpmnist1000_nofisher/` — long-horizon stress
and the attenuation on/off variant at 1,000 tasks.
`cifar100_main/`, `cifar100_lambda/`, `cifar100_buffer/`,
`cifar100_layers/`, `cifar100_proj/` — CIFAR-100 main + ablations.
`tinyimagenet_main/`, `imagenet_r_main/` — the two additional Class-IL
benchmarks (the latter under domain shift).
