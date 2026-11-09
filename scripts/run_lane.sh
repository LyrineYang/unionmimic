#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run_dir="$1"
shift
mkdir -p "$run_dir"
bash scripts/run_training.sh night --run "$run_dir" "$@" 2>&1 | tee -a "$run_dir/supervisor.log"
