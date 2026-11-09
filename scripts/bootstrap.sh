#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
mkdir -p work/downloads work/tools work/upstreams work/checkpoints work/logs

# An existing measured runtime is verified, never silently reinstalled.
if [[ -x "$UNIONMIMIC_VENV_DIR/bin/python" && -f config/runtime-lock.json ]]; then
    python scripts/check_runtime.py
    echo 'Existing isolated runtime verified. To rebuild, set UNIONMIMIC_VENV_DIR to a new directory.'
    exit 0
fi

check_hash() {
    local expected="$1" artifact="$2"
    printf '%s  %s\n' "$expected" "$artifact" | sha256sum --check --status
}
download() {
    local url="$1" artifact="$2" expected="$3"
    if [[ -f "$artifact" ]] && check_hash "$expected" "$artifact"; then
        return
    fi
    curl -fL --retry 2 --connect-timeout 20 --max-time 1800 "$url" -o "${artifact}.partial"
    check_hash "$expected" "${artifact}.partial"
    mv -- "${artifact}.partial" "$artifact"
}

if [[ ! -x "$UNIONMIMIC_VENV_DIR/bin/python" ]]; then
    download 'https://github.com/astral-sh/python-build-standalone/releases/download/20241002/cpython-3.8.20%2B20241002-x86_64-unknown-linux-gnu-install_only.tar.gz' \
        work/downloads/python38.tar.gz 285e141c36f88b2e9357654c5f77d1f8fb29cc25132698fe35bb30d787f38e87
    if [[ ! -x work/tools/python/bin/python3.8 ]]; then
        tar -xzf work/downloads/python38.tar.gz -C work/tools
    fi
    work/tools/python/bin/python3.8 -c 'import sys; assert sys.version_info[:3] == (3, 8, 20)'
    work/tools/python/bin/python3.8 -m venv "$UNIONMIMIC_VENV_DIR"
fi

if [[ ! -d work/upstreams/WM-Craftnet/.git ]]; then
    git clone https://github.com/sharpa-robotics/WM-Craftnet.git work/upstreams/WM-Craftnet
    git -C work/upstreams/WM-Craftnet checkout --detach a1e825acaf50ade298e30b0b13c6671627af3a1e
fi
if [[ "$(git -C work/upstreams/WM-Craftnet rev-parse HEAD)" != a1e825acaf50ade298e30b0b13c6671627af3a1e ]]; then
    echo 'Upstream revision mismatch; inspect existing work before changing it.' >&2
    exit 1
fi
bash scripts/apply_upstream_patches.sh

download https://developer.nvidia.com/isaac-gym-preview-4 \
    work/downloads/IsaacGym_Preview_4_Package.tar.gz c51d77ac1144b91b356f439c7313d8d59468147e37187ebffe7cdbc3154fdb8f
if [[ ! -d work/isaacgym/python ]]; then
    tar -xzf work/downloads/IsaacGym_Preview_4_Package.tar.gz -C work
fi
download https://huggingface.co/SharpaIT/WM-Craftnet/resolve/main/wm_craftnet_set_z.pth \
    work/checkpoints/wm_craftnet_set_z.pth 3215205fe1fad48e97f5745ab90bb2377d53012c1a0d2634edf186699ccab7f7

python -m pip install --index-url https://pypi.org/simple pip==24.3.1 setuptools==69.5.1 wheel==0.45.1
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118 \
    'torch==2.1.0+cu118' 'torchvision==0.16.0+cu118' 'numpy==1.22.2'
if [[ -f config/requirements-runtime.lock.txt ]]; then
    python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118 \
        -r config/requirements-runtime.lock.txt
else
    python -m pip install --index-url https://pypi.org/simple -r config/requirements-runtime.txt
fi
python -m pip install --no-deps --no-build-isolation -e work/isaacgym/python

download https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.5.tar.gz \
    work/downloads/pytorch3d-v0.7.5.tar.gz 335c51933485e8bf44b88dbf07d0106c05362da8830cef3968983314db0645d5
if [[ ! -d work/upstreams/pytorch3d-0.7.5 ]]; then
    tar -xzf work/downloads/pytorch3d-v0.7.5.tar.gz -C work/upstreams
fi
CUDA_HOME=/usr/local/cuda-11.8 CUB_HOME=/usr/local/cuda-11.8/include FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.0 \
    python -m pip install --no-deps --no-build-isolation work/upstreams/pytorch3d-0.7.5
python -m pip check
python -m pip freeze > work/logs/environment-freeze.txt
echo 'Dependencies installed. Run bash scripts/run_step1.sh for actual GPU/simulator validation.'
