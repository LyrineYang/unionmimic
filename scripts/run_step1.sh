#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH="${UNIONMIMIC_ROOT}/work/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}"
exec python scripts/verify_step1.py "$@"
