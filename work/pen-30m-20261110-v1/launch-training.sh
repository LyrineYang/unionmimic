#!/usr/bin/env bash
set -e
cd /home/junqiy/projects/unionmimic
source scripts/env.sh
task="$1"; gpu="$2"; candidate="$3"
dir="$PWD/work/pen-30m-20261110-v1"
remain=$(python - "$dir/authorization.json" <<'PY'
import json,sys,time
print(int(json.load(open(sys.argv[1]))['deadline_unix']-time.time()))
PY
)
test "$remain" -ge 540
run="$PWD/work/runs/sharpa-pen-30m-task$task-20261110-v1"
timeout "${remain}s" python -u scripts/train_pen_quick.py --gpu "$gpu" --task "$task" --candidate "$candidate" \
    --authorization "$dir/authorization.json" --run "$run"
