"""Validate separately authorized short comparisons without changing PPO's recipe."""
from pathlib import Path


def validate_plan(plan, recipe, run, seed, now, root):
    assert plan['approved_by_user'] is True, 'Training plan has not been authorized'
    assert plan['recipe'] == recipe, 'PPO recipe differs from the reviewed reference'
    assert (Path(root)/plan['run']).resolve() == Path(run).resolve(), 'Run does not match plan'
    assert plan['seed'] == seed, 'Seed does not match plan'
    hours = float(plan['hours'])
    if plan.get('scope') == 'quick_sm_ppo_comparison':
        assert 0 < hours <= 2, 'Quick comparison must finish within two hours'
        assert now + hours*3600 <= plan['authorization_deadline_unix'], 'Authorization expires before completion'
        assert plan.get('automatic_resume') is False
    else:
        assert 8 <= hours <= 12
    return hours
