#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
exec python -m tensorboard.main --logdir "$1" --host 127.0.0.1 --port 6006 --reload_interval 15
