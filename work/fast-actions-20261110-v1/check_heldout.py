"""Evaluate the first checkpoint meeting the locked development gate within the same hour."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

base = Path(__file__).resolve().parent
repo = base.parent.parent
lane = sys.argv[1]
plan = json.loads((base / f'{lane}.json').read_text())
run = repo / plan['run']
manifest = json.loads((run / 'manifest.json').read_text())
status = json.loads((run / 'status.json').read_text())
assert status['status'] == 'completed'
receipt_path = run / 'heldout-status.json'
assert not receipt_path.exists()
gate = dict(rotation_rad=25., failure_rate=.10, axis_deviation_rad=.35)


def passes(result):
    return (
        result['mean_rotation_rad'] >= gate['rotation_rad']
        and result['failure_rate'] <= gate['failure_rate']
        and result['mean_axis_deviation_rad'] <= gate['axis_deviation_rad']
    )


def finish(state, **details):
    data = dict(status=state, gate=gate, seed=plan['heldout_seed'], unix=time.time(), **details)
    tmp = receipt_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(receipt_path)
    print(json.dumps(data), flush=True)


events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
candidates = [
    event for event in events
    if event['kind'] == 'evaluation' and event['passed']
    and event['spin_sign'] == plan['spin_sign']
    and event['seed'] == plan['development_seed'] and passes(event)
]
if not candidates:
    finish('development_gate_not_met')
    sys.exit(0)

selected = min(candidates, key=lambda event: event['checkpoint_elapsed_seconds'])
if lane.startswith('sm-'):
    name = selected['name']
    checkpoint = run / 'ckpt' / ('final.pt' if name == 'evaluation-final' else f'actor-{name[len("evaluation-"):]}.pt')
else:
    checkpoint = run / f"actor-{selected['label']}.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert digest == selected['checkpoint_sha256']
remaining = manifest['deadline_unix'] - time.time() - 5
selected_details = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                        development_checkpoint_seconds=selected['checkpoint_elapsed_seconds'])
if remaining < 150:
    finish('insufficient_original_time', **selected_details)
    sys.exit(0)

result_path = run / 'evaluation-heldout.json'
try:
    with (run / 'evaluation-heldout.log').open('w') as log:
        subprocess.run([
            sys.executable, 'scripts/evaluate_sharpa_leap.py', '--gpu', str(plan['gpu']),
            '--checkpoint', str(checkpoint), '--out', str(result_path), '--seed', str(plan['heldout_seed']),
        ], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
    finish('heldout_evaluation_failed', reason=repr(error), **selected_details)
    raise
result = json.loads(result_path.read_text())
assert result['checkpoint_sha256'] == digest and result['seed'] == plan['heldout_seed']
assert result['spin_sign'] == plan['spin_sign'] and result['episodes'] == len(result['rows']) == 512
finish('confirmed' if result['passed'] and passes(result) else 'heldout_gate_not_met',
       report=str(result_path), **selected_details)
