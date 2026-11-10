"""Verify all four immutable GPU preflight receipts before launching the pairings."""
import hashlib
import json
import math
import time
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parent.parent
receipts = {}
for plan_path in sorted(root.glob('*.json')):
    if plan_path.name == 'debug-passed.json':
        continue
    plan = json.loads(plan_path.read_text())
    pilot = repo / plan['pilot']
    manifest = json.loads((pilot / 'manifest.json').read_text())
    status = json.loads((pilot / 'status.json').read_text())
    is_sm = plan_path.name.startswith('sm-')
    acceptance = json.loads((pilot / ('operator-acceptance.json' if is_sm else 'acceptance.json')).read_text())
    evaluation = json.loads((pilot / 'evaluation.json').read_text())
    checkpoint = pilot / ('ckpt/operator.pt' if is_sm else 'actor.pt')
    assert status['status'] == 'completed' and status['gpu'] == plan['gpu']
    assert manifest['cli']['seed'] == plan['seed'] and manifest['cli']['spin_sign'] == plan['spin_sign']
    assert all(hashlib.sha256((repo / file).read_bytes()).hexdigest() == digest
               for file, digest in manifest['sources'].items())
    assert acceptance['passed']
    if is_sm:
        assert acceptance['model_updates'] == 10 and acceptance['actor_updates'] == 1
        assert acceptance['critic_optimizer_steps'] == 20 and acceptance['used_actor_updates'] >= 1
        assert acceptance['frozen_model_unchanged'] and acceptance['actor']['finite']
    else:
        assert acceptance['iterations'] == 2 and acceptance['optimizer_steps'] == 40
        assert acceptance['recollected_policy_iterations'] >= 1
    assert evaluation['passed'] and evaluation['spin_sign'] == plan['spin_sign']
    assert evaluation['seed'] == plan['development_seed'] and evaluation['episodes'] == len(evaluation['rows']) == 512
    assert all(math.isfinite(v) for row in evaluation['rows'] for v in row.values())
    assert evaluation['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert evaluation['reward_max_error'] <= .002
    receipts[plan_path.stem] = {
        'gpu': plan['gpu'], 'pilot': plan['pilot'], 'spin_sign': plan['spin_sign'],
        'checkpoint_sha256': evaluation['checkpoint_sha256'], 'reward_max_error': evaluation['reward_max_error'],
        'acceptance_seconds': acceptance['seconds'], 'evaluation_seconds': evaluation['seconds'],
    }

assert len(receipts) == 4 and {item['gpu'] for item in receipts.values()} == set(range(4))
tmp = root / 'debug-passed.tmp'
tmp.write_text(json.dumps(dict(passed=True, checked_unix=time.time(), lanes=receipts), indent=2) + '\n')
tmp.replace(root / 'debug-passed.json')
print((root / 'debug-passed.json').read_text(), end='')
