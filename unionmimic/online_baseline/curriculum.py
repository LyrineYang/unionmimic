"""Explicit task curriculum persistence; no simulator imports required."""
import math

FIELDS = ('gravity_curriculum_current_max_angle_deg', 'gravity_curriculum_upgrade_count',
          'gravity_curriculum_full_run_ratio', 'gravity_curriculum_outcome_write_idx',
          'gravity_curriculum_outcome_count', 'action_conservative_scale')


def capture_curriculum(env):
    return dict(schema=1, values={k:getattr(env,k) for k in FIELDS},
                outcomes=env.gravity_curriculum_outcomes.detach().cpu().clone())


def restore_curriculum(env, saved):
    if saved.get('schema') != 1 or set(saved.get('values', {})) != set(FIELDS):
        raise ValueError('Missing or incompatible curriculum checkpoint')
    values = saved['values']; outcomes = saved['outcomes']
    if outcomes.shape != env.gravity_curriculum_outcomes.shape:
        raise ValueError('Curriculum window changed')
    if not all(math.isfinite(float(v)) for v in values.values()):
        raise ValueError('Nonfinite curriculum')
    if not bool(outcomes.isfinite().all()) or not bool(((outcomes>=0)&(outcomes<=1)).all()):
        raise ValueError('Invalid curriculum outcomes')
    size = len(outcomes)
    if not 0 <= values['gravity_curriculum_outcome_write_idx'] < size:
        raise ValueError('Invalid curriculum index')
    if not 0 <= values['gravity_curriculum_outcome_count'] <= size:
        raise ValueError('Invalid curriculum count')
    if not env.gravity_curriculum_start_max_angle_deg <= values['gravity_curriculum_current_max_angle_deg'] <= env.gravity_curriculum_max_angle_deg:
        raise ValueError('Curriculum angle out of task bounds')
    for key, value in values.items():
        setattr(env, key, value)
    env.gravity_curriculum_outcomes.copy_(outcomes.to(env.device))
