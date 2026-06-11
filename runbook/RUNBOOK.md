# PA-CL Experiment Runbook

This runbook is the **single source of truth** for the experiment plan
and the order in which we execute the project on the GPU.

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
                                                     plasticity.png/.pdf,
                                                     rank_traj.png/.pdf,
                                                     gate.{json,txt}, raw.json
```

## Phase 0 — Engineering smoke test (offline, ~30 s on CPU)

Verifies the entire pipeline (data, model, all 3 methods, CBP post-step,
PA-CL two-backward path, Fisher accumulator, rank diagnostics, aggregator,
plot generation, gate evaluation) without needing a GPU or internet
access. Uses `fake_pmnist` so it works in sandboxed environments.

```bash
cd code
pip install -r requirements.txt   # or use the GPU host's pre-built env
for M in erm cbp pacl; do
  python scripts/run_experiment.py \
      --config configs/dryrun_fake.yaml --method $M --seed 0
done
python scripts/aggregate_poc.py --out_dir runbook/dryrun_fake \
       --methods erm cbp pacl
```

Pass criteria: all 3 runs complete; `runbook/dryrun_fake/_aggregate/`
contains `table.md`, `gate.json`, `plasticity.png`, `rank_traj.png`.
(The numbers themselves are meaningless — the fake dataset is noise.)

## Phase 1 — Consolidated PoC (GPU, ~2 hours on a single A100)

This is **the** go/no-go gate. Runs Continual-Permuted-MNIST with 200
tasks for 3 methods (ERM, CBP, PA-CL) on 3 seeds = 9 cells. Produces
the paper's headline plasticity-decay figure, the effective-rank
trajectory figure, and the ACC/BWT/plasticity-drop table.

```bash
cd code
bash scripts/run_poc.sh                # runs all 9 cells + aggregator
# Watch progress:
ls runbook/poc_cpmnist200/*/seed*/summary.json
# Inspect gate:
cat runbook/poc_cpmnist200/_aggregate/gate.txt
```

### Gate (strict, locked by the user)

PASS requires ALL of:
- **G1** — `erank(PA-CL, last) / erank(ERM, last) >= 1.5` on the
  penultimate hidden layer (`hidden.h2`).
- **G2** — ERM plasticity drop >= 5 pp AND PA-CL plasticity drop
  <= 0.5 × ERM's drop.
- **G3** — ACC(PA-CL) - ACC(ERM) >= 1.5 pp AND no overlap in
  mean ± std bars.
- **G4** — BWT(PA-CL) >= BWT(ERM) - 2 pp.

PARTIAL: ACC gain in (0, 1.5 pp) OR rank ratio in [1.2, 1.5).
FAIL:    ACC gain < 0 OR rank ratio < 1.2.

PARTIAL → tune `lam` / probe layers / EMA `beta`, re-run.
FAIL    → pivot mechanism per Section 8 of the research proposal
          (F1 long-horizon Permuted-MNIST, F2 stronger projection,
          F6 log barrier instead of squared hinge).

## Phase 2 — Main-table sweep (~80 GPU-hours on A100, ~7 days wall-clock)

Only after Phase 1 PASS. Six benchmarks (S-CIFAR-100, S-TinyImageNet,
S-ImageNet-R, CORe50, CPMNIST-200, 5-Datasets) × 10+ baselines × 5
seeds. Adds: ResNet-18 / ViT-S backbones, ER / DER++ / EWC / SI / GPM
/ LwF / Flashbacks / Progressive-NC baselines, paired-bootstrap p-values
with Bonferroni correction.

## Phase 3 — Ablation matrix (~60 GPU-hours)

See `paper/sections/ablations.tex` for the eight ablation axes.

## Phase 4 — PT-ViT runs on Split-ImageNet-R (~30 GPU-hours)

L2P, DualPrompt, CODA-Prompt, FM-LoRA, DuPt under the modern PT-CL
protocol with ViT-S/16.

## Phase 5 — Failure-case analyses (~20 GPU-hours)

Long-horizon stress (1000 tasks), high class-overlap, OOD-task
injection, Theorem-1 assumption audit on 2-layer ReLU.

## Logs

`dryrun_fake/`     — populated by Phase 0 (engineering only).
`poc_cpmnist200/`  — populated by Phase 1.
Other directories will be populated by later phases.