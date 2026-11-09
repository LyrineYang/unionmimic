"""Validate actual native pilot outputs; does not certify learning convergence."""
import argparse
import json
import math
from pathlib import Path

import torch


def check_finite(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), 'Nonfinite saved tensor'
    elif isinstance(value, dict):
        for child in value.values():
            check_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value), 'Nonfinite scalar'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    learning = run / 'learning'
    cfg = json.loads((learning / 'args.json').read_text())
    expected = dict(envs=4096, replay=1000, horizon=16, starts=4096,
                    wm_steps=10, wm_batch=2048, wm_k=8, wm_hidden='1024,1024,1024',
                    wm_only=300, utd=0.4, warmup=64, hours=0.75)
    for key, value in expected.items():
        assert cfg[key] == value, (key, cfg[key], value)
    assert cfg['wm_init'] is None and cfg['collect_policy'] is None
    actor = torch.load(learning / 'ckpt/actor_final.pt', map_location='cpu', weights_only=False)
    model = torch.load(learning / 'models_final.pt', map_location='cpu', weights_only=False)
    check_finite(actor)
    check_finite(model)
    assert actor['update'] >= 320, 'Did not complete 20 actor updates after native model warm-up'
    assert list(actor['hidden']) == [512, 512, 256]
    assert list(model['cfg']['hidden']) == [1024, 1024, 1024]
    rows = [json.loads(line) for line in (learning / 'log.jsonl').read_text().splitlines()]
    check_finite(rows)
    assert any(row['update'] > 300 and row['gn'] > 0 for row in rows), 'No logged actor gradient'
    result = dict(engineering_passed=True, learning_convergence_verified=False,
                  updates=actor['update'], actor_updates=actor['update'] - 300,
                  environment_transitions=actor['env_steps'], training_seconds=actor['wall_s'],
                  exact_default_recipe_except_duration_and_output=True,
                  saved_tensors_finite=True, rows=len(rows), latest=rows[-1],
                  resumable_optimizer_replay=False)
    (run / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
