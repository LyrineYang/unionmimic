"""Validate committed stage artifacts before explicit supervisor recovery."""
import json
from pathlib import Path
import h5py
import numpy as np
from .common import file_digest
from .data_contracts import audit_episode

STAGES=('sm','evaluate','collect','merge','prepare','calibrate','calibrate-extend')


def option(params,key):
    return params[params.index(key)+1]


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_completed_stage(phase,params):
    out=Path(option(params,'--output')).resolve()
    result=dict(phase=phase,output=str(out))
    if phase=='evaluate':
        report=read_json(out)
        checkpoint=Path(option(params,'--checkpoint')).resolve()
        count=int(option(params,'--envs'))
        assert report['checkpoint_sha256']==file_digest(checkpoint), 'Evaluation belongs to another checkpoint'
        assert Path(report['checkpoint']).resolve()==checkpoint
        assert report['episodes']==count and report['seed']==int(option(params,'--seed'))
        assert len(report['returns'])==len(report['lengths'])==count
        assert np.isfinite(report['returns']).all() and np.isfinite(report['mean_return'])
        assert all(0<n<=500 for n in report['lengths'])
        assert abs(np.mean(report['returns'])-report['mean_return'])<1e-3
        assert 0<=report['falls']<=report['failures']<=count and 0<=report['timeouts']<=count
        assert abs(report['failure_rate']-report['failures']/count)<1e-6
        result.update(episodes=count,checkpoint_sha256=report['checkpoint_sha256'])
    elif phase in ('collect','merge'):
        report=read_json(out/'collection.json')
        with h5py.File(out/'episodes.h5','r') as f:
            assert f.attrs['complete'] and len(f)==report['episodes']>0
            assert (out/'config.yaml').is_file()
            if phase=='collect':
                assert f.attrs['source_checkpoint_sha256']==file_digest(option(params,'--checkpoint'))
                assert report['parameters_unchanged'] and report['terminal_observations_preserved']
            counts=dict(transitions=0,failures=0,falls=0,timeouts=0)
            for key in f:
                g=f[key]
                data={k:g[k][:] for k in g}
                audit_episode(data,f.attrs['lower_limits'],f.attrs['upper_limits'])
                assert g.attrs['split'] in ('train','valid')
                counts['transitions']+=len(data['reward'])-1
                for name,field in [('failures','failure'),('falls','fall'),('timeouts','timeout')]:
                    counts[name]+=int(data[field][-1])
            assert all(report[k]==v for k,v in counts.items())
        result.update(episodes=report['episodes'],**counts)
    elif phase=='sm':
        report=read_json(out/'completed.json')
        assert all(report[k] for k in ('actor_changed','world_model_unchanged','failure_head_unchanged',
                                      'normalization_unchanged','critic_frozen_during_actor_update'))
        import torch
        saved=torch.load(str(out/'last.pt'),map_location='cpu')
        assert saved['sm_state']['step']==report['step'] and saved['sm_report']==report
        result['step']=report['step']
    elif phase=='prepare':
        report=read_json(out/'recovery.json')
        assert report['seeded_replay_exact'] and report['full_episode_history_used']
        if report.get('posterior_mode')=='recomputed_from_full_history':
            assert '--recompute-state' in params
            assert report['recorded_posterior_reproduced'] is False
            assert report['max_deter_error'] is None and report['max_logit_error'] is None
            assert report['max_embed_error']<2e-4
        else:
            assert max(report[k] for k in ('max_deter_error','max_logit_error','max_embed_error'))<2e-4
        assert report['source_checkpoint_sha256']==file_digest(option(params,'--checkpoint'))
        import torch
        saved=torch.load(str(out/'cache.pt'),map_location='cpu')
        assert saved['metadata']['dataset_sha256']==report['dataset_sha256']
        assert saved['metadata']['state_model_sha256']==report['state_model_sha256']
        assert set(saved['train']['episode_id'].tolist()).isdisjoint(saved['valid']['episode_id'].tolist())
    elif phase.startswith('calibrate'):
        completed=read_json(out/'completed.json')
        report=read_json(out/'validation.json')
        import torch
        saved=torch.load(str(out/'best.pt'),map_location='cpu')
        assert saved['calibration_report']==report
        assert saved['calibration_step']<=completed['updates'] and (out/'last.pt').is_file()
        result['passed']=report['passed']
    else:
        raise ValueError('No recovery validator for '+phase)
    return result
