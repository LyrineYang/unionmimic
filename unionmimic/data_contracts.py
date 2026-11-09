"""CPU-only replay checks, shared by collection and interrupted-run recovery."""
import numpy as np


def audit_episode(data,lower,upper):
    n=len(data['reward'])
    assert n>=2
    assert data['is_first'][0] and not data['is_first'][1:].any()
    assert not data['done'][:-1].any() and data['done'][-1]
    assert (np.diff(data['progress'])==1).all(), 'Reset boundary leaked into an episode'
    assert np.array_equal(data['done'],data['failure'] | data['timeout'])
    expected_a=.8*data['applied_command'][1:]+.2*data['last_actions'][:-1]
    action_error=float(np.abs(expected_a-data['executed_action'][1:]).max())
    expected_targets=np.clip(data['targets'][:-1]+data['scale'][1:]*data['executed_action'][1:],lower,upper)
    target_error=float(np.abs(expected_targets-data['targets'][1:]).max())
    assert action_error<2e-6 and target_error<2e-6,(action_error,target_error)
    assert np.allclose(data['last_actions'],data['executed_action'],atol=1e-7)
    for key,value in data.items():
        assert np.isfinite(value).all(), 'Nonfinite data: '+key
    return action_error,target_error
