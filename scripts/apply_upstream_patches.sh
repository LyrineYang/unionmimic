#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
project_root="$PWD"
for patch_file in "$project_root"/patches/*.patch; do
    if git -C work/upstreams/WM-Craftnet apply --reverse --check "$patch_file" 2>/dev/null; then
        continue
    fi
    git -C work/upstreams/WM-Craftnet apply --check "$patch_file"
    git -C work/upstreams/WM-Craftnet apply "$patch_file"
done
