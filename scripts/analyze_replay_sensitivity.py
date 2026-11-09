"""Analyze saved physical branches using one fixed posterior for all models.

The simulator prefixes reproduce exposed physical starts but camera/posterior
noise need not repeat. Reusing each branch's noisy posterior would confound the
model action difference; hold the recorded pre-perturbation posterior fixed.
"""
import argparse
import json
from pathlib import Path
import unionmimic.runtime
import torch
import numpy as np
from unionmimic.models import load_world_model,CalibratedHeads
from unionmimic.sensitivity import imagined_branch
from unionmimic.common import write_json,digest,file_digest


def main(args):
    p=args.folder;checkpoint=args.checkpoint
    tape=torch.load(str(p/'replay-tape.pt'),map_location='cpu')
    assert tape['checkpoint_sha256']==file_digest(checkpoint)
    base=torch.load(str(p/'replay-record/physical.pt'),map_location='cpu')
    wm,saved,cfg=load_world_model(args.config,checkpoint,args.gpu);device=next(wm.parameters()).device
    heads=CalibratedHeads(wm).to(device).eval();heads.failure.load_state_dict(saved['failure_head'])
    heads.failure_log_weight.fill_(saved['failure_log_weight']);heads.requires_grad_(False)
    before=digest(wm)
    initial={k:v.to(device) for k,v in base['start']['latent'].items()}
    last=base['start']['last_action'].to(device);applied=base['physical']['applied'].to(device)
    directions=tape['directions'].to(device)
    null=[json.loads((p/n/'replay.json').read_text()) for n in ('replay-record','replay-null-1','replay-null-2')]
    noise=np.ptp(np.array([n['returns'] for n in null]),axis=0)
    grad_samples=[]
    for seed in (101,102,103):
        delta=torch.zeros_like(last,requires_grad=True)
        prediction=imagined_branch(wm,heads,initial,last,applied,delta,64,seed)
        grad_samples.append(torch.autograd.grad(prediction['reward'].sum(),delta)[0])
    grad=torch.stack(grad_samples).mean(0);rows=[]
    for d in (0,1):
        pairs={};predictions={}
        for sign in (-1,1):
            branch=torch.load(str(p/('replay-d%d-s%d'%(d,sign))/'physical.pt'),map_location='cpu')
            for key in ('root','dof','last_action'):
                assert torch.equal(branch['start'][key],base['start'][key]),'Physical start differs: '+key
            pairs[sign]=branch['physical'];predictions[sign]=[]
            for seed in (101,102,103):
                with torch.no_grad():
                    pred=imagined_branch(wm,heads,initial,last,applied,sign*.03*directions[d],64,seed)
                predictions[sign].append({k:v.cpu() for k,v in pred.items()})
        for lane in range(tape['envs']):
            neg,pos=pairs[-1],pairs[1]
            delta_real=float(pos['reward'][:,lane].sum()-neg['reward'][:,lane].sum())
            model_deltas=[float(predictions[1][i]['reward'][:,lane].sum()-predictions[-1][i]['reward'][:,lane].sum()) for i in range(3)]
            mapping=max(float((pairs[s]['executed'][:,lane]-predictions[s][0]['executed'][:,lane]).abs().max()) for s in (-1,1))
            eligible=bool(pos['mask'][:,lane].all() and neg['mask'][:,lane].all() and mapping<2e-5
                          and abs(delta_real)>max(3*noise[lane],.01))
            rows.append(dict(direction=d,lane=lane,physical_delta=delta_real,
                physical_rotation_delta=float(pos['spin'][:,lane].sum()-neg['spin'][:,lane].sum()),
                model_delta_mean=float(np.mean(model_deltas)),model_delta_mc_samples=model_deltas,
                gradient_projection=float((grad[lane]*directions[d,lane]).sum()*.06),
                null_range=float(noise[lane]),action_mapping_max_error=mapping,eligible=eligible))
    eligible=[r for r in rows if r['eligible']]
    assert before==digest(wm)
    report=dict(status='small_pilot_inconclusive_not_gate_pass',states=4,directions=2,amplitude=.03,
        model_start='one fixed recorded pre-perturbation posterior for every branch',model_mc_per_branch=192,
        physical_null_ranges=noise.tolist(),eligible=len(eligible),comparisons=rows,
        gradient_sign_matches=sum(r['physical_delta']*r['gradient_projection']>0 for r in eligible),
        model_sign_matches=sum(r['physical_delta']*r['model_delta_mean']>0 for r in eligible),
        optimizer_steps=0,weights_unchanged=True,
        limitations=['Four starts only; no positive/negative causal verdict from this pilot.',
                     'Physical null replay has nonzero noise; only effects above a predefined bound retained.',
                     'Monte Carlo delta estimates are noisy; this is not a validated derivative of true contact dynamics.',
                     'Full contact/slip stratification and held-out action coverage remain gates before long SM training.'])
    write_json(p/'action-response-fixed-start.json',report)
    print({k:v for k,v in report.items() if k!='comparisons'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3);main(p.parse_args())
