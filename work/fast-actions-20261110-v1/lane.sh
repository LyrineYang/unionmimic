#!/usr/bin/env bash
set -e -o pipefail
cd /home/junqiy/projects/unionmimic
source scripts/env.sh
method="$1"
direction="$2"
gpu="$3"
phase="$4"
seed=20261104
sign=1
[[ "$direction" != negz ]] || sign=-1
lane="$method-$direction"
batch=work/fast-actions-20261110-v1
pilot="$PWD/work/runs/sharpa-1h-$lane-operator-20261110-v1"
run="$PWD/work/runs/sharpa-1h-$lane-20261110-v1"
exec >"$batch/$lane-$phase.log" 2>&1
if [[ "$phase" == debug ]]; then
    if [[ "$method" == sm ]]; then
        timeout 6m python -u scripts/train_sharpa_leap.py --run "$pilot" --gpu "$gpu" --seed "$seed" --spin-sign "$sign" --engineering-check
        checkpoint="$pilot/ckpt/operator.pt"
    else
        timeout 6m python -u scripts/pilot_sharpa_ppo.py --run "$pilot" --gpu "$gpu" --seed "$seed" --spin-sign "$sign" --iterations 2
        checkpoint="$pilot/actor.pt"
    fi
    timeout 6m python -u scripts/evaluate_sharpa_leap.py --gpu "$gpu" --checkpoint "$checkpoint" --out "$pilot/evaluation.json"
elif [[ "$phase" == learn ]]; then
    test -s "$batch/debug-passed.json"
    if [[ "$method" == sm ]]; then
        timeout 3600s python -u scripts/train_sharpa_leap.py --run "$run" --gpu "$gpu" --seed "$seed" --spin-sign "$sign" --hours 1 \
          --operator-evidence "$pilot/operator-acceptance.json" --quick-plan "$batch/$lane.json"
    else
        timeout 3600s python -u scripts/pilot_sharpa_ppo.py --run "$run" --gpu "$gpu" --seed "$seed" --spin-sign "$sign" \
          --approved-plan "$batch/$lane.json"
    fi
else
    exit 2
fi
