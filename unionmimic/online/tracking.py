"""Stable tracking identity across segments, distinct for each study arm."""
from pathlib import Path


def tracking_run_id(run, arm):
    run = Path(run)
    # Both long-training lanes use a child named learner. That basename alone
    # would merge SM and PPO into the same remote run.
    name = run.parent.parent.name + '-' + run.parent.name if run.name == 'learner' else run.name
    return name.replace('_', '-')
