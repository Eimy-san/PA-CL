#!/usr/bin/env bash
# Consolidated PoC sweep: CPMNIST-200 x {ERM, CBP, PA-CL} x 3 seeds.
# Runs the 9 cells sequentially, then calls the aggregator which
# computes the gate verdict per Section 5 of the research proposal.
#
# Usage (from repo root):
#   bash scripts/run_poc.sh                   # full 200-task sweep
#   bash scripts/run_poc.sh tiny              # 4-task dry-run (CPU OK)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-full}"
case "$MODE" in
  full)  CFG="configs/poc_cpmnist200.yaml" ; SEEDS="0 1 2" ;;
  tiny)  CFG="configs/poc_cpmnist_tiny.yaml" ; SEEDS="0" ;;
  *)     echo "unknown mode: $MODE (use full|tiny)" >&2 ; exit 2 ;;
esac

METHODS=("erm" "cbp" "pacl")

echo "[poc] config=$CFG  seeds='$SEEDS'  methods=${METHODS[*]}"

for METHOD in "${METHODS[@]}"; do
  for SEED in $SEEDS; do
    echo
    echo "==============================================================="
    echo "[poc] launching  method=$METHOD  seed=$SEED"
    echo "==============================================================="
    python scripts/run_experiment.py \
      --config "$CFG" \
      --method "$METHOD" \
      --seed   "$SEED"
  done
done

# Aggregate.
OUT_DIR=$(python - <<PY
import yaml
with open("$CFG") as f: c = yaml.safe_load(f)
print(c.get("out_dir", "runbook/poc_cpmnist200"))
PY
)
echo
echo "[poc] all cells done. Aggregating from $OUT_DIR ..."
python scripts/aggregate_poc.py --out_dir "$OUT_DIR" --methods "${METHODS[@]}"