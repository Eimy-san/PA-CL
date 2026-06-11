#!/usr/bin/env bash
# Phase-2 main-table sweep: Class-IL Split-CIFAR-100, ResNet-18.
# Runs 8 methods x 5 seeds = 40 cells. Aggregates at the end.
#
# Method registry (8 cells per seed):
#   Baselines: erm, ewc, er, derpp, agem
#   PA-CL:     pacl (=PA-CL+ERM), pacl_er, pacl_derpp
#
# Wall-time estimate on a single 4090: ~6-8 hours total.
#   ERM/EWC/AGEM: ~10 min/seed each
#   ER/DER++:     ~15 min/seed each
#   PA-CL:        ~15-20 min/seed (Fisher overhead)
#   PA-CL + ER:   ~20-25 min/seed (Fisher + buffer overhead)
#   Total per seed: ~115 min. Five seeds: ~10 h. Reduce to 3 seeds for
#   tighter wall budget (recommended for initial confirmation, then
#   extend to 5 if results are PASS-grade).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-full}"
case "$MODE" in
  full)  CFG="configs/cifar100_main.yaml" ; SEEDS="0 1 2 3 4" ;;
  quick) CFG="configs/cifar100_main.yaml" ; SEEDS="0 1 2" ;;
  smoke) CFG="configs/cifar100_main.yaml" ; SEEDS="0" ;;
  *) echo "unknown mode: $MODE (use full|quick|smoke)" >&2; exit 2 ;;
esac

METHODS=("erm" "ewc" "er" "derpp" "agem" "pacl" "pacl_er" "pacl_derpp")

echo "[cifar-sweep] config=$CFG  seeds='$SEEDS'  methods=${METHODS[*]}"

for METHOD in "${METHODS[@]}"; do
  for SEED in $SEEDS; do
    OUT_FILE="runbook/cifar100_main/${METHOD}/seed${SEED}/summary.json"
    if [[ -f "$OUT_FILE" ]]; then
      echo "[cifar-sweep] SKIP (already done): $METHOD seed=$SEED"
      continue
    fi
    echo
    echo "==============================================================="
    echo "[cifar-sweep] launching  method=$METHOD  seed=$SEED"
    echo "==============================================================="
    python3 scripts/run_experiment.py \
      --config "$CFG" \
      --method "$METHOD" \
      --seed   "$SEED"
  done
done

OUT_DIR=$(python3 - <<PY
import yaml
with open("$CFG") as f: c = yaml.safe_load(f)
print(c.get("out_dir", "runbook/cifar100_main"))
PY
)
echo
echo "[cifar-sweep] all cells done. Aggregating from $OUT_DIR ..."
python3 scripts/aggregate_poc.py --out_dir "$OUT_DIR" --methods "${METHODS[@]}" \
        --penult_layer "avgpool_tap"