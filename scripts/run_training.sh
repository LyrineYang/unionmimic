#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
bash scripts/apply_upstream_patches.sh
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH="${UNIONMIMIC_ROOT}/work/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
exec python -m unionmimic.cli "$@"
