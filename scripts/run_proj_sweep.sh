#!/usr/bin/env bash
# PA-CL Fisher-null projection on/off ablation.
# 2 arms x 3 seeds = 6 cells.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="configs/cifar100_proj.yaml"
SEEDS=(0 1 2)

echo "[proj-sweep] config=$CFG  seeds=${SEEDS[*]}"

for SEED in "${SEEDS[@]}"; do
  for ARM in on off; do
    TAG="pacl_er_proj_${ARM}"
    OUT_FILE="runbook/cifar100_proj/${TAG}/seed${SEED}/summary.json"
    if [[ -f "$OUT_FILE" ]]; then
      echo "[proj-sweep] SKIP (already done): $TAG seed=$SEED"
      continue
    fi
    EXTRA=""
    if [[ "$ARM" == "off" ]]; then
      EXTRA="--disable_projection"
    fi
    echo
    echo "============================================================"
    echo "[proj-sweep] launching  arm=$ARM  seed=$SEED  tag=$TAG"
    echo "============================================================"
    python3 scripts/run_experiment.py \
      --config "$CFG" \
      --seed   "$SEED" \
      --tag    "$TAG" \
      $EXTRA
  done
done

echo
echo "[proj-sweep] all cells done."
# Use the existing method-vs-method aggregator (proj_on / proj_off
# behave like two distinct "methods").
python3 scripts/aggregate_poc.py --out_dir runbook/cifar100_proj \
        --methods pacl_er_proj_on pacl_er_proj_off \
        --penult_layer avgpool_tap