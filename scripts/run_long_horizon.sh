#!/usr/bin/env bash
# Long-horizon CPMNIST sweep: 1000 tasks x {ERM, CBP, PA-CL} x 3 seeds.
# Resume-safe (skips cells whose summary.json already exists).
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="configs/poc_cpmnist1000.yaml"
SEEDS="0 1 2"
METHODS=("erm" "cbp" "pacl")

echo "[long-horizon] config=$CFG seeds='$SEEDS' methods=${METHODS[*]}"

for METHOD in "${METHODS[@]}"; do
  for SEED in $SEEDS; do
    OUT_FILE="runbook/poc_cpmnist1000/${METHOD}/seed${SEED}/summary.json"
    if [[ -f "$OUT_FILE" ]]; then
      echo "[long-horizon] SKIP (already done): $METHOD seed=$SEED"
      continue
    fi
    echo
    echo "==============================================================="
    echo "[long-horizon] launching method=$METHOD seed=$SEED"
    echo "==============================================================="
    python3 scripts/run_experiment.py \
      --config "$CFG" \
      --method "$METHOD" \
      --seed   "$SEED"
  done
done

echo
echo "[long-horizon] all cells done. Aggregating..."
python3 scripts/aggregate_poc.py \
  --out_dir runbook/poc_cpmnist1000 \
  --methods "${METHODS[@]}" \
  --penult_layer "hidden.h2"