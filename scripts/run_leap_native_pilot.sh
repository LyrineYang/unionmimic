#!/usr/bin/env bash
# Bounded native pilot only. This entry point cannot launch a long run.
set -euo pipefail
UNIONMIMIC_PILOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$UNIONMIMIC_PILOT_ROOT"
source scripts/env.sh
UNIONMIMIC_PILOT_RUN="${1:?absolute new run directory required}"
UNIONMIMIC_PILOT_GPU="${2:?physical GPU index required}"
[[ "$UNIONMIMIC_PILOT_RUN" == "$UNIONMIMIC_PILOT_ROOT/work/runs/"* ]]
[[ "$UNIONMIMIC_PILOT_GPU" =~ ^[0-3]$ ]]
[[ ! -e "$UNIONMIMIC_PILOT_RUN" ]]
mkdir -p "$UNIONMIMIC_PILOT_RUN"
export CUDA_VISIBLE_DEVICES="$UNIONMIMIC_PILOT_GPU"
python - "$UNIONMIMIC_PILOT_RUN" "$UNIONMIMIC_PILOT_GPU" <<'PY'
import hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=Path(sys.argv[1]);plan=json.loads(Path('config/leap-native-validation-20261103.plan.json').read_text())
up=root/plan['upstream']['path']
assert subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip()==plan['upstream']['commit']
for path,value in plan['upstream']['source_sha256'].items():
    assert hashlib.sha256((up/path).read_bytes()).hexdigest()==value,path
for path,value in plan['orchestration_sha256'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==value,path
runtime=json.loads((root/'work/analysis/leap-native-20261103/runtime.json').read_text())
assert runtime['passed']
runtime_files=['work/runtime/leap/pyproject.toml','work/runtime/leap/uv.lock',
               'work/analysis/leap-native-20261103/installed-packages.json']
manifest=dict(started_unix=time.time(),gpu=int(sys.argv[2]),plan=plan,
              training_command=['scripts/wm/train_yaw.py','--out',str(out/'learning'),'--hours','0.75','--device','cuda:0'],
              long_training_approved=False,from_scratch=True,auto_resume=False,
              runtime=runtime,runtime_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in runtime_files})
snapshot=out/'launch_sources';snapshot.mkdir()
for path in plan['orchestration_sha256']:
    (snapshot/Path(path).name).write_bytes((root/path).read_bytes())
for path in runtime_files:
    (snapshot/Path(path).name).write_bytes((root/path).read_bytes())
manifest['training_deadline_unix']=manifest['started_unix']+3900
manifest['total_deadline_unix']=manifest['started_unix']+4500
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'status.json').write_text(json.dumps(dict(status='running',phase='train',updated_unix=time.time()),indent=2)+'\n')
PY
UNIONMIMIC_PILOT_PHASE=train
trap 'UNIONMIMIC_PILOT_RC=$?; python - "$UNIONMIMIC_PILOT_RUN" "$UNIONMIMIC_PILOT_PHASE" "$UNIONMIMIC_PILOT_RC" <<"PY"
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);rc=int(sys.argv[3]);tmp=p/"status.json.tmp"
state="completed" if rc==0 else "stopped" if rc in (130,143) else "failed"
tmp.write_text(json.dumps(dict(status=state,phase=sys.argv[2],exit_code=rc,updated_unix=time.time()),indent=2)+"\n")
tmp.replace(p/"status.json")
PY
exit "$UNIONMIMIC_PILOT_RC"' EXIT
timeout --signal=TERM --kill-after=30s 3900s bash scripts/leap_python.sh \
  work/upstreams/sm-reference-audit/scripts/wm/train_yaw.py \
  --out "$UNIONMIMIC_PILOT_RUN/learning" --hours 0.75 --device cuda:0 \
  > "$UNIONMIMIC_PILOT_RUN/train.log" 2>&1
UNIONMIMIC_PILOT_PHASE=validate
bash scripts/leap_python.sh scripts/validate_leap_native_pilot.py \
  --run "$UNIONMIMIC_PILOT_RUN" > "$UNIONMIMIC_PILOT_RUN/validation.log" 2>&1
UNIONMIMIC_PILOT_PHASE=evaluate
timeout --signal=TERM --kill-after=30s 600s bash scripts/leap_python.sh \
  work/upstreams/sm-reference-audit/scripts/eval/eval_yaw.py \
  --policy "$UNIONMIMIC_PILOT_RUN/learning/ckpt/actor_final.pt" \
  --out "$UNIONMIMIC_PILOT_RUN/evaluation.json" --device cuda:0 \
  > "$UNIONMIMIC_PILOT_RUN/evaluation.log" 2>&1
UNIONMIMIC_PILOT_PHASE=finished
