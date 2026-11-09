"""Compare physical and model action responses. No parameter updates."""
import argparse
import time
from pathlib import Path
from unionmimic.runtime import make_runtime, close_runtime
import torch
import numpy as np
from scipy.stats import spearmanr
from unionmimic.models import CalibratedHeads
from unionmimic.sensitivity import snapshot,restore,physical_branch,imagined_branch
from unionmimic.common import digest,file_digest,write_json,save_checkpoint
from unionmimic.simulation_worker import run_isolated_simulation


def main(args):
    args.checkpoint=args.checkpoint.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    env,player,cfg,obs=make_runtime(args.envs,args.gpu,args.seed,args.checkpoint,
        overrides=['task.env.gravityCurriculum.maxAngleDeg=1.0'])
    wm=env.wm_adapter.world_model;wm.requires_grad_(False)
    saved=torch.load(str(args.checkpoint),map_location='cpu')
    heads=CalibratedHeads(wm).to(env.device).eval()
    heads.failure.load_state_dict(saved['failure_head']);heads.failure_log_weight.fill_(saved['failure_log_weight'])
    heads.requires_grad_(False)
    before=(digest(wm),digest(player.model));rows=[];null_rows=[];details=[]
    elapsed=0;began=time.monotonic()
    for group in range(args.groups):
        # Return to an unperturbed continuation after each group's branches.
        for _ in range(args.warmup if group==0 else args.spacing):
            obs,_,_,_=player.env_step(env,player.get_action(obs,True))
            elapsed+=1
        initial={k:v.clone() for k,v in env.wm_adapter._wm_latent.items()}
        last_action=env.last_actions.clone();start=snapshot(env)
        start_valid=~env.reset_buf.bool().clone()
        commands=[]
        for _ in range(4):
            command=player.get_action(obs,True);commands.append(command.clone())
            obs,_,_,_=player.env_step(env,command)
        continuation=snapshot(env)
        continuation_obs={k:v.clone() for k,v in obs.items() if torch.is_tensor(v)}
        nulls=[]
        for _ in range(args.null_repeats):
            restore(env,start);nulls.append(physical_branch(env,player,commands))
        baseline=nulls[0]
        real_return=torch.stack([(n['reward']*n['mask']).sum(0) for n in nulls])
        noise=(real_return-real_return.mean(0)).abs().max(0).values
        # Non-saturated command coordinates avoid conflating a clamp with an
        # incorrect local derivative. Seed the directions independently.
        gen=torch.Generator(device=env.device);gen.manual_seed(args.seed+group+1000)
        directions=torch.randn(args.directions,args.envs,22,generator=gen,device=env.device)
        safe=(commands[0].abs()<.8) & (baseline['applied'][0].abs()<.8)
        directions*=safe[None]
        directions/=directions.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        delta=torch.zeros_like(last_action,requires_grad=True)
        model0=imagined_branch(wm,heads,initial,last_action,baseline['applied'],delta,args.mc,101+group)
        grad=torch.autograd.grad(model0['reward'].sum(),delta)[0]
        for lane in range(args.envs):
            null_rows.append(dict(group=group,lane=lane,valid=bool(start_valid[lane]),
                return_noise_max=float(noise[lane]),returns=real_return[:,lane].tolist(),
                joint_restore_max_error=max(float((n['joint'][:,lane]-baseline['joint'][:,lane]).abs().max()) for n in nulls),
                h1_reward_error=float(model0['reward'][0,lane]-baseline['reward'][0,lane]),
                h4_reward_error=float(model0['reward'][3,lane]-baseline['reward'][3,lane])))
        for direction_id,direction in enumerate(directions):
            for amplitude in args.amplitudes:
                pairs={}
                for sign in (-1,1):
                    perturbation=sign*amplitude*direction
                    varied=[x.clone() for x in commands];varied[0]=(varied[0]+perturbation).clamp(-1,1)
                    restore(env,start);physical=physical_branch(env,player,varied)
                    with torch.no_grad():
                        prediction=imagined_branch(wm,heads,initial,last_action,baseline['applied'],perturbation,args.mc,101+group)
                    pairs[sign]=(physical,prediction)
                for lane in range(args.envs):
                    neg,pneg=pairs[-1];pos,ppos=pairs[1]
                    # Stop interpreting the derivative when a branch crosses
                    # a reset. Report that boundary separately instead.
                    complete=bool(neg['mask'][:,lane].all() and pos['mask'][:,lane].all())
                    real=float((pos['reward'][:,lane]*pos['mask'][:,lane]).sum()-(neg['reward'][:,lane]*neg['mask'][:,lane]).sum())
                    predicted=float(ppos['reward'][:,lane].sum()-pneg['reward'][:,lane].sum())
                    projected=float((grad[lane]*direction[lane]).sum()*2*amplitude)
                    mapping=max(float((p['executed'][:,lane]-m['executed'][:,lane]).abs().max()) for p,m in pairs.values())
                    rows.append(dict(group=group,lane=lane,direction=direction_id,amplitude=amplitude,
                        valid_start=bool(start_valid[lane]),complete=complete,real_delta_return=real,
                        model_delta_return=predicted,gradient_delta_return=projected,
                        real_delta_rotation=float((pos['spin'][:,lane]*pos['mask'][:,lane]).sum()-(neg['spin'][:,lane]*neg['mask'][:,lane]).sum()),
                        restore_noise_bound=float(2*noise[lane]),action_mapping_max_error=mapping))
                details.append(dict(group=group,direction=direction_id,amplitude=amplitude,
                    negative={k:v.cpu() for k,v in pairs[-1][0].items()},positive={k:v.cpu() for k,v in pairs[1][0].items()}))
        restore(env,continuation);obs=continuation_obs
        print('sensitivity group',group+1,'rows',len(rows),flush=True)
    eligible=[r for r in rows if r['valid_start'] and r['complete'] and r['action_mapping_max_error']<2e-5
              and abs(r['real_delta_return'])>max(3*r['restore_noise_bound'],.01)]
    def correlation(key):
        if len(eligible)<3:return None
        value=float(spearmanr([r['real_delta_return'] for r in eligible],[r[key] for r in eligible]).statistic)
        return value if np.isfinite(value) else None
    report=dict(mode='read_only_action_sensitivity',checkpoint_sha256=file_digest(args.checkpoint),
        gpu=args.gpu,seed=args.seed,envs=args.envs,groups=args.groups,rows=rows,null_replays=null_rows,
        eligible_comparisons=len(eligible),total_comparisons=len(rows),
        model_rank_correlation=correlation('model_delta_return'),gradient_rank_correlation=correlation('gradient_delta_return'),
        model_sign_agreement=(float(np.mean([np.sign(r['real_delta_return'])==np.sign(r['model_delta_return']) for r in eligible])) if eligible else None),
        gradient_sign_agreement=(float(np.mean([np.sign(r['real_delta_return'])==np.sign(r['gradient_delta_return']) for r in eligible])) if eligible else None),
        optimizer_steps=0,seconds=time.monotonic()-began,
        interpretation='diagnostic_only; repeated directions within a start are dependent; no causal pass from pooled significance',
        limitations=['Contact-solver hidden state cannot be fully restored; null replay noise is measured.',
                     'Only effects above three times measured restoration bound are used in summary.',
                     'Legacy checkpoint has no calibrated rotation head; only physical rotation delta is available.',
                     'Fixed open-loop continuation isolates action response; this is not closed-loop policy improvement.'])
    assert before==(digest(wm),digest(player.model))
    write_json(args.output/'sensitivity.json',report)
    save_checkpoint(args.output/'branches.pt',details)
    print({k:v for k,v in report.items() if k not in ('rows','null_replays')},flush=True)
    close_runtime(env)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--seed',type=int,default=62001);p.add_argument('--envs',type=int,default=16)
    p.add_argument('--groups',type=int,default=4);p.add_argument('--directions',type=int,default=8)
    p.add_argument('--amplitudes',type=float,nargs='+',default=[.01,.03,.05])
    p.add_argument('--warmup',type=int,default=32);p.add_argument('--spacing',type=int,default=60)
    p.add_argument('--null-repeats',type=int,default=3);p.add_argument('--mc',type=int,default=16)
    run_isolated_simulation(main,p.parse_args())
