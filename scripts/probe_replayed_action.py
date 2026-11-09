"""Rebuild PhysX contact history in a fresh process before each action branch.

This is a measurement control, not a training run. Matching seeds is not assumed
to imply matching physical starts; both the starts and null outcomes are saved.
"""
import argparse
from pathlib import Path
from unionmimic.runtime import make_runtime,close_runtime
import torch
from unionmimic.common import save_checkpoint,write_json,digest,file_digest
from unionmimic.models import CalibratedHeads
from unionmimic.sensitivity import imagined_branch
from unionmimic.collect import terminal_flags
from unionmimic.simulation_worker import run_isolated_simulation


def main(args):
    for k in ('checkpoint','tape','output'):setattr(args,k,getattr(args,k).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    tape=None if args.record else torch.load(str(args.tape),map_location='cpu')
    if tape:
        assert tape['checkpoint_sha256']==file_digest(args.checkpoint)
        assert (args.seed,args.envs,args.warmup)==(tape['seed'],tape['envs'],len(tape['prefix']))
    env,player,cfg,obs=make_runtime(args.envs,args.gpu,args.seed,args.checkpoint,
        overrides=['task.env.gravityCurriculum.maxAngleDeg=1.0'])
    wm=env.wm_adapter.world_model;wm.requires_grad_(False)
    before=(digest(wm),digest(player.model));prefix=[]
    for t in range(args.warmup):
        # get_action consumes the same RNG stream even when executing a tape.
        inferred=player.get_action(obs,True)
        command=inferred if tape is None else tape['prefix'][t].to(env.device)
        prefix.append(command.cpu());obs,_,_,_=player.env_step(env,command)
    initial={k:v.clone() for k,v in env.wm_adapter._wm_latent.items()}
    last_action=env.last_actions.clone()
    start=dict(root=env.root_state_tensor.clone(),dof=env.dof_state.clone(),
        valid=~env.reset_buf.bool().clone(),latent=initial,last_action=last_action)
    directions=None if tape is None else tape['directions'].to(env.device)
    commands=[];rows=[];live=start['valid'].clone()
    for t in range(4):
        inferred=player.get_action(obs,True)
        command=inferred if tape is None else tape['commands'][t].to(env.device).clone()
        if t==0 and args.sign:
            command=(command+args.sign*args.amplitude*directions[args.direction]).clamp(-1,1)
        commands.append(command.cpu())
        obs,_,_,_=player.env_step(env,command)
        flags=terminal_flags(env)
        rows.append(dict(reward=env.rew_buf.clone(),joint=env.last_obs_buf[:,:22].clone(),
            spin=env.last_spin_delta_axis.clone()/20,failure=flags['failure'].clone(),mask=live.clone(),
            applied=env.applied_command.clone(),executed=env.executed_command.clone()))
        live &= ~env.reset_buf.bool()
    physical={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
    if tape is None:
        gen=torch.Generator(device=env.device);gen.manual_seed(args.seed+1000)
        directions=torch.randn(2,args.envs,22,generator=gen,device=env.device)
        safe=(commands[0].to(env.device).abs()<.8)&(physical['applied'][0].abs()<.8)
        directions*=safe;directions/=directions.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        tape=dict(seed=args.seed,envs=args.envs,prefix=prefix,commands=commands,directions=directions.cpu(),
                  start={k:v.cpu() for k,v in start.items() if torch.is_tensor(v)},
                  checkpoint_sha256=file_digest(args.checkpoint))
        save_checkpoint(args.tape,tape)
    saved=torch.load(str(args.checkpoint),map_location='cpu')
    heads=CalibratedHeads(wm).to(env.device).eval();heads.failure.load_state_dict(saved['failure_head'])
    heads.failure_log_weight.fill_(saved['failure_log_weight']);heads.requires_grad_(False)
    # Compare each branch from its own measured posterior. Start mismatch is
    # separately reported; these values must not be pooled as exact derivatives.
    delta=torch.zeros_like(last_action,requires_grad=True)
    model=imagined_branch(wm,heads,initial,last_action,physical['applied'],delta,16,101)
    grad=torch.autograd.grad(model['reward'].sum(),delta)[0]
    start_error={k:float((start[k]-tape['start'][k].to(env.device)).abs().max()) for k in ('root','dof')}
    report=dict(seed=args.seed,envs=args.envs,sign=args.sign,direction=args.direction,amplitude=args.amplitude,
        start_max_errors=start_error,valid_start=start['valid'].tolist(),complete=physical['mask'].all(0).tolist(),
        returns=(physical['reward']*physical['mask']).sum(0).tolist(),
        rotations=(physical['spin']*physical['mask']).sum(0).tolist(),model_returns=model['reward'].sum(0).detach().tolist(),
        projected_gradients=torch.einsum('dba,ba->db',directions,grad).tolist(),
        action_mapping_max_error=float((model['executed']-physical['executed']).abs().max()),
        optimizer_steps=0,weights_unchanged=before==(digest(wm),digest(player.model)))
    assert report['weights_unchanged']
    write_json(args.output/'replay.json',report)
    save_checkpoint(args.output/'physical.pt',dict(start=start,physical=physical))
    print(report,flush=True);close_runtime(env)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tape',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--record',action='store_true');p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--seed',type=int,default=62001);p.add_argument('--envs',type=int,default=4)
    p.add_argument('--warmup',type=int,default=32);p.add_argument('--sign',type=int,choices=[-1,0,1],default=0)
    p.add_argument('--direction',type=int,choices=[0,1],default=0);p.add_argument('--amplitude',type=float,default=.03)
    run_isolated_simulation(main,p.parse_args())
