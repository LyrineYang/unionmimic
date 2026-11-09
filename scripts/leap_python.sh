#!/usr/bin/env bash
# Isolated LEAP interpreter; never activate its libraries in the legacy shell.
set -euo pipefail
UNIONMIMIC_LEAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$UNIONMIMIC_LEAP_ROOT/scripts/env.sh"
UNIONMIMIC_LEAP_SYSROOT="$UNIONMIMIC_LEAP_ROOT/work/tools/leap-sysroot/x86_64-conda-linux-gnu/sysroot"
UNIONMIMIC_LEAP_PYBASE="$UNIONMIMIC_LEAP_ROOT/work/lafan1/env"
export VIRTUAL_ENV="$UNIONMIMIC_LEAP_ROOT/work/envs/leap-native-py311"
export PYTHONPATH="$UNIONMIMIC_LEAP_ROOT/work/upstreams/sm-reference-audit/src"
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$UNIONMIMIC_LEAP_ROOT/work/cache/leap-native"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
mkdir -p "$XDG_CACHE_HOME" "$WARP_CACHE_PATH"
# --library-path applies to this interpreter, not unrelated shell subprocesses.
unset LD_PRELOAD
export LD_LIBRARY_PATH="$UNIONMIMIC_LEAP_PYBASE/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
exec "$UNIONMIMIC_LEAP_SYSROOT/lib64/ld-linux-x86-64.so.2" \
  --library-path "$UNIONMIMIC_LEAP_SYSROOT/lib64:$LD_LIBRARY_PATH" \
  "$VIRTUAL_ENV/bin/python" "$@"
