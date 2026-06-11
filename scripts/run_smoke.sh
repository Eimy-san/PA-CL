#!/usr/bin/env bash
# Engineering smoke test for PA-CL.
# Runs the smoke config end-to-end and asserts the pass criteria.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/run_experiment.py --config configs/smoke_pmnist.yaml

R="runbook/smoke_pmnist/R_seed0.npy"
S="runbook/smoke_pmnist/summary_seed0.json"
T="runbook/smoke_pmnist/rank_traj_seed0.json"

for f in "$R" "$S" "$T"; do
  if [[ ! -f "$f" ]]; then
    echo "[smoke FAIL] missing artefact: $f"
    exit 1
  fi
done

python - <<'PY'
import json, numpy as np
R = np.load("runbook/smoke_pmnist/R_seed0.npy")
assert R.shape == (5, 5), f"R shape {R.shape}, expected (5,5)"
with open("runbook/smoke_pmnist/rank_traj_seed0.json") as f:
    traj = json.load(f)
assert len(traj) == 5, f"rank_traj len {len(traj)}, expected 5"
# Monotone non-decreasing per layer
layers = list(traj[0].keys())
for lname in layers:
    vals = [d[lname] for d in traj]
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:])), \
        f"rank floor for {lname} not monotone: {vals}"
print("[smoke PASS] R shape ok, rank-floor monotone, artefacts present.")
PY