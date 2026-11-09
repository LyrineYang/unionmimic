"""Explicit branch provenance and exact rational collection scheduling."""
from fractions import Fraction
import torch


def collection_steps(cfg, cycle):
    if 'utd' not in cfg:
        return cfg['collect_steps']
    ratio = Fraction(str(cfg['utd']))
    if not 0 < ratio <= 1:
        raise ValueError('UTD must be in (0, 1] for collection-first scheduling')
    n = int(cycle) - int(cfg.get('branch_initial_cycles', 0))
    if n < 0:
        raise ValueError('Counters predate the branch origin')
    # Difference of integer floors: .4 gives 2,3,2,3...; no float drift.
    return ((n + 1) * ratio.denominator // ratio.numerator
            - n * ratio.denominator // ratio.numerator)


def check_finite(tree):
    if torch.is_tensor(tree):
        if tree.is_floating_point() and not bool(torch.isfinite(tree).all()):
            raise ValueError('Nonfinite branch checkpoint')
    elif isinstance(tree, dict):
        for value in tree.values(): check_finite(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree: check_finite(value)


def validate_branch(saved, cfg, task_constants):
    check_finite(saved)
    if saved['task_constants'] != task_constants:
        raise ValueError('Branch changes physical task constants')
    if saved['replay']['capacity'] != cfg['replay_steps'] or saved['replay']['nenv'] != cfg['envs']:
        raise ValueError('Branch replay layout mismatch')
    allowed = {'failure_fraction','critic_norm_source','horizon','deadline_unix','launch_authorization'}
    for key, value in saved['config'].items():
        if key in cfg and key not in allowed and cfg[key] != value:
            raise ValueError('Unplanned branch configuration change: '+key)
    if cfg['branch_initial_cycles'] != saved['learner']['counts']['cycles']:
        raise ValueError('Branch cycle origin mismatch')
    if cfg['horizon'] not in (8,16):
        raise ValueError('Unreviewed screening horizon')
    if str(cfg['utd']) not in ('0.125','0.4'):
        raise ValueError('Unreviewed screening UTD')
