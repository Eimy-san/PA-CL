#!/usr/bin/env bash
# Regenerate ALL PA-CL paper figures from the committed runbook logs.
# Requires: matplotlib, numpy; the runbook data directories.
set -e
cd "$(dirname "$0")/.."
python scripts/regenerate_all_figures.py
python scripts/theorem_diagnostic.py
