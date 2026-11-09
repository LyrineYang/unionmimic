#!/usr/bin/env bash
# Source from any working directory. Keep runtime and caches in this project.
UNIONMIMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIONMIMIC_ROOT
if [[ -f "${UNIONMIMIC_ROOT}/config/network.local.env" ]]; then
    source "${UNIONMIMIC_ROOT}/config/network.local.env"
fi
if [[ -n "${UNIONMIMIC_PROXY_URL:-}" ]]; then
    export HTTP_PROXY="$UNIONMIMIC_PROXY_URL" HTTPS_PROXY="$UNIONMIMIC_PROXY_URL" ALL_PROXY="$UNIONMIMIC_PROXY_URL"
    export http_proxy="$UNIONMIMIC_PROXY_URL" https_proxy="$UNIONMIMIC_PROXY_URL" all_proxy="$UNIONMIMIC_PROXY_URL"
fi
if [[ -n "${UNIONMIMIC_NO_PROXY:-}" ]]; then
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${UNIONMIMIC_NO_PROXY}"
    export no_proxy="$NO_PROXY"
fi
# One interpreter per project. Do not inherit another venv's Python search path.
export UNIONMIMIC_VENV_DIR="${UNIONMIMIC_VENV_DIR:-${UNIONMIMIC_ROOT}/.venv}"
export VIRTUAL_ENV="$UNIONMIMIC_VENV_DIR"
unset PYTHONHOME
case "$PATH" in
    "${UNIONMIMIC_VENV_DIR}/bin:"*) ;;
    *) export PATH="${UNIONMIMIC_VENV_DIR}/bin:${PATH}" ;;
esac
export PIP_CACHE_DIR="${UNIONMIMIC_ROOT}/work/cache/pip"
export XDG_CACHE_HOME="${UNIONMIMIC_ROOT}/work/cache"
export HF_HOME="${UNIONMIMIC_ROOT}/work/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${UNIONMIMIC_ROOT}/work/cache/torch_extensions"
export TMPDIR="${UNIONMIMIC_ROOT}/work/tmp"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
source "${UNIONMIMIC_ROOT}/config/wandb.env"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export LD_LIBRARY_PATH="${UNIONMIMIC_ROOT}/work/tools/python/lib:${UNIONMIMIC_ROOT}/work/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${UNIONMIMIC_ROOT}:${UNIONMIMIC_ROOT}/work/upstreams/WM-Craftnet"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
mkdir -p "$WANDB_DIR"
