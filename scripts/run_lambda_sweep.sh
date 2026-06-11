#!/usr/bin/env bash
# PA-CL lambda ablation: 5 lambda values, 1 seed, PA-CL+ER on Split-CIFAR-100.
# Each cell is written to runbook/cifar100_lambda/lam_<value>/seed0/.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="configs/cifar100_lambda.yaml"
LAMBDAS=("0.0" "0.01" "0.1" "1.0" "10.0")
SEED=0

echo "[lambda-sweep] config=$CFG  lambdas=${LAMBDAS[*]}  seed=$SEED"

for L in "${LAMBDAS[@]}"; do
  TAG="lam_${L}"
  OUT_FILE="runbook/cifar100_lambda/${TAG}/seed${SEED}/summary.json"
  if [[ -f "$OUT_FILE" ]]; then
    echo "[lambda-sweep] SKIP (already done): $TAG"
    continue
  fi
  echo
  echo "============================================================"
  echo "[lambda-sweep] launching  lambda=$L  tag=$TAG"
  echo "============================================================"
  python3 scripts/run_experiment.py \
    --config "$CFG" \
    --seed   "$SEED" \
    --lam    "$L" \
    --tag    "$TAG"
done

echo
echo "[lambda-sweep] all cells done. Building summary table..."
python3 scripts/aggregate_lambda.py --out_dir runbook/cifar100_lambda