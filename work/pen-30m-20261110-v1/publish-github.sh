#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
pen=work/pen-30m-20261110-v1
fast=work/fast-actions-20261110-v1
quick_ppo=work/runs/sharpa-paired-ppo-quick-20261109-v1
sm_baseline=work/runs/geometry-sm-baseline-20261109-v1
paths=()

include_existing() {
    local path
    for path in "$@"; do
        if [[ -s "$path" ]]; then paths+=("$path"); fi
    done
}

check_terminal() {
    python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text())[sys.argv[2]]
if state not in {'completed', 'failed', 'stopped'}:
    sys.exit(f'{path} 当前为 {state}，请在运行结束后重试。')
print(f'{path}: {state}', file=sys.stderr)
print(state)
PY
}

check_digest() {
    python3 - "$1" "$2" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint, evaluation = map(Path, sys.argv[1:])
expected = json.loads(evaluation.read_text())['checkpoint_sha256']
digest = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != expected:
    sys.exit(f'评估与权重的 SHA-256 不一致：{evaluation} / {checkpoint}')
PY
}

case "${1:-}" in
    source)
        message='Sync Sharpa fast-training fixes and validation evidence'
        references=("$pen"/calibrated-retarget/*.pt)
        [[ ${#references[@]} -eq 13 ]] || { echo '转笔参考张量应为 13 个' >&2; exit 1; }
        paths=(
            README.md
            docs/{DEMOS,GEOMETRY_AND_DEMO_BACKUP_20261109,MAINLINE,PEN_ADAPTATION_AND_QUEUE_20261109,SM_PPO_VALIDATION_20261109,SM_REPRODUCTION,STATUS,TECHNICAL_HANDOFF_20261110,TRAINING}.md
            scripts/{calibrate_pen_quick_lead,check_pen_quick_physics,evaluate_pen_policy,evaluate_sharpa_leap,pilot_sharpa_ppo,prepare_leap_sharpa,train_pen_quick,train_sharpa_leap}.py
            tests/test_leap_sharpa_loop.py
            unionmimic/leap_sharpa/{model,pen_model}.py
            unionmimic/online_baseline/simulator.py
            unionmimic/{pen_quick,pen_simulator}.py
            "$fast"/{debug-passed,sm-posz,sm-negz,ppo-posz,ppo-negz}.json
            "$fast"/{check_debug.py,check_heldout.py,lane.sh}
            "$pen"/{authorization.json,launch-training.sh,publish-github.sh,stopped-posz-checkpoints.sha256}
            "$pen"/stopped-{sm,ppo}-posz-{before,after}.json
            "$pen"/{lead-high-depen,lead-fixed-grip-175}/physics-task0-seed202612{40,41}.json
            "$pen"/calibrated-retarget/manifest.json "${references[@]}"
            work/runs/sharpa-pen-30m-task{2,3}-20261110-v1/manifest.json
            "$quick_ppo"/{manifest,status,acceptance,best-evaluation,evaluation-000054,evaluation-000107,evaluation-000159,evaluation-final}.json
            "$quick_ppo"/{events.jsonl,actor-final.pt}
            "$sm_baseline"/{status,evaluation-final}.json
        )
        ;;
    results)
        message='Publish evaluated Sharpa single-axis and multi-axis pen runs'
        for run in work/runs/sharpa-pen-30m-task{2,3}-20261110-v1; do
            state=$(check_terminal "$run/status.json" stage)
            paths+=("$run"/{manifest.json,status.json,environment.json,events.jsonl})
            if [[ "$state" == completed || -s "$run/evaluation-final.json" ]]; then
                check_digest "$run/final.pt" "$run/evaluation-final.json"
            fi
            if [[ -s "$run/evaluation-zero-residual.json" ]]; then
                check_digest "$run/final.pt" "$run/evaluation-zero-residual.json"
            fi
            if [[ -s "$run/final.pt" ]]; then
                paths+=("$run/final.pt")
            else
                include_existing "$run/latest.pt"
            fi
            include_existing "$run"/{evaluation-final.json,evaluation-zero-residual.json,evaluation-final.log,evaluation-zero-residual.log}
        done
        ;;
    corner)
        message='Publish the one-hour Sharpa corner-block comparison evidence'
        for run in work/runs/sharpa-1h-{sm,ppo}-{posz,negz}-20261110-v1; do
            state=$(check_terminal "$run/status.json" status)
            if [[ "$run" == *-negz-* && "$state" == completed && ! -s "$run/heldout-status.json" ]]; then
                echo "等待 $run/heldout-status.json" >&2
                exit 1
            fi
            paths+=("$run"/{manifest.json,status.json,events.jsonl})
            include_existing "$run"/{acceptance,best-evaluation,evaluation-final,heldout-status,evaluation-heldout}.json
        done
        ;;
    *) echo "用法: bash $0 {source|results|corner} [--dry-run]" >&2; exit 2 ;;
esac

for path in "${paths[@]}"; do
    [[ -s "$path" ]] || { echo "缺少文件：$path" >&2; exit 1; }
    if (( $(stat -c %s -- "$path") > 104857600 )); then
        echo "文件超过 GitHub 的 100 MiB 上限：$path" >&2
        exit 1
    fi
done
if [[ "${2:-}" == --dry-run ]]; then
    printf '%s\n' "${paths[@]}"
    exit 0
fi
[[ $# -eq 1 ]] || exit 2
[[ "$(git symbolic-ref --quiet --short HEAD)" == main ]] || { echo '请在 main 上运行' >&2; exit 1; }
[[ "$(git remote get-url origin)" == https://github.com/LyrineYang/unionmimic.git &&
   "$(git remote get-url --push origin)" == https://github.com/LyrineYang/unionmimic.git ]] || {
    echo 'origin 与预期的 LyrineYang/unionmimic 不一致' >&2; exit 1;
}
git diff --cached --quiet || { echo '暂存区已有改动，请先检查 git diff --cached' >&2; exit 1; }
git fetch origin main
git merge-base --is-ancestor FETCH_HEAD HEAD || { echo '远端 main 已前进，请先检查分支历史' >&2; exit 1; }
git add -f -- "${paths[@]}"
git diff --cached --check
git diff --cached --stat
if ! git diff --cached --quiet; then git commit -m "$message"; fi
git push origin HEAD:refs/heads/main
remote_head=$(git ls-remote origin refs/heads/main | cut -f1)
[[ "$remote_head" == "$(git rev-parse HEAD)" ]]
printf 'GitHub main 已同步到 %s\n' "$remote_head"
