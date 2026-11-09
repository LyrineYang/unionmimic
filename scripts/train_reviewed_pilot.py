"""Bounded pilot entry point. The checked-in approval template denies updates.

All numeric defaults live in the reviewed manifest. This script never launches
another arm, grows a budget, resumes the old run, or enables W&B.
"""
import argparse
import json
import time
import os
import traceback
from pathlib import Path
from unionmimic.common import ROOT,write_json,save_checkpoint,digest
from unionmimic.review_gate import ReviewGate


def progress(args, **fields):
    write_json(args.output/'status.json',dict(pid=os.getpid(),arm=args.arm,
        status='running',updated_unix=time.time(),**fields))


def ppo(args):
    from unionmimic.runtime import make_runtime,close_runtime
    import torch
    from omegaconf import OmegaConf
    from unionmimic.ppo import make_ppo,make_critic,sample_policy,collect_ppo_block,ppo_minibatch,update_ppo_epoch,pilot_checkpoint
    gate=ReviewGate(args.manifest,args.approval,'ppo');gate.verify_inputs();gate()
    m=gate.manifest;c=m['ppo_pilot'];checkpoint=ROOT/m['checkpoint']
    began=time.monotonic();deadline=began+c['max_seconds']
    env,player,cfg,obs=make_runtime(c['envs'],args.gpu,c['seed'],checkpoint,
        overrides=['task.env.gravityCurriculum.maxAngleDeg=1.0'])
    saved=torch.load(str(checkpoint),map_location='cpu');wm=env.wm_adapter.world_model
    # Agent owns only the actor network. WM inference remains detached/frozen.
    wm.requires_grad_(False)
    agent=make_ppo(OmegaConf.to_container(cfg,resolve=True),saved,env.device,args.output,
        c['envs'],c['horizon'],min(c['minibatch'],c['envs']*c['horizon']),c['actor_lr'],False,gate)
    critic=make_critic(wm,env.device)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=c['critic_lr'])
    before=(digest(wm),digest(agent.model.running_mean_std),digest(agent.model.value_mean_std))
    actor_initial=digest(agent.model);critic_initial=digest(critic)
    frames=0;completed=0;skipped=0;actor_steps=0;critic_steps=0;stop='pilot_epoch_limit'
    with open(args.output/'metrics.jsonl','x',buffering=1) as log:
        for epoch in range(1,c['max_epochs']+1):
            if time.monotonic()>=deadline:stop='pilot_time_cap';break
            progress(args,epoch=epoch,phase='collect',physical_frames=frames,actor_steps=actor_steps)
            blocks=[];n=0;started=time.monotonic()
            while n<c['batch_frames']:
                if time.monotonic()>=deadline:break
                batch,obs,invalid=collect_ppo_block(env,player,agent,critic,obs,c['horizon'])
                blocks.append(batch);n+=len(batch['reward']);frames+=c['horizon']*c['envs'];skipped+=invalid
            if time.monotonic()>=deadline:stop='pilot_time_cap';break
            data={k:torch.cat([b[k] for b in blocks]) for k in blocks[0]}
            sampling=time.monotonic()-started;started=time.monotonic();agent.epoch_num=epoch
            result=update_ppo_epoch(agent,critic,critic_opt,data,c['minibatch'],c['mini_epochs'],deadline,gate,False)
            if result is None:stop='pilot_time_cap';break
            actor_steps=agent.optimizer_steps;critic_steps+=result['critic_steps']
            assert before==(digest(wm),digest(agent.model.running_mean_std),digest(agent.model.value_mean_std))
            row=dict(epoch=epoch,physical_frames=frames,valid_batch=n,excluded_reset_transitions=skipped,
                actor_steps=actor_steps,**result,sample_seconds=sampling,
                update_seconds=time.monotonic()-started,seconds=time.monotonic()-began)
            row.update(actor_changed=digest(agent.model)!=actor_initial,
                       critic_changed=digest(critic)!=critic_initial,frozen_modules_unchanged=True)
            assert row['actor_changed'] and row['critic_changed']
            log.write(json.dumps(row,allow_nan=False)+'\n');print(row,flush=True)
            for key,value in row.items():
                if isinstance(value,(int,float)):agent.writer.add_scalar('pilot/'+key,value,frames)
            agent.writer.flush();progress(args,phase='checkpoint',**row)
            bundle=pilot_checkpoint(saved,agent,critic,critic_opt,epoch,frames,gate.sha,row)
            # New isolated checkpoints; the source and original best are untouched.
            save_checkpoint(args.output/'last.pt',bundle)
            if epoch==1 or epoch%c['save_every_epochs']==0 or epoch==c['max_epochs']:
                save_checkpoint(args.output/('epoch-%03d.pt'%epoch),bundle)
            completed=epoch
            if time.monotonic()>=deadline:stop='pilot_time_cap';break
    write_json(args.output/'completed.json',dict(status=stop,epochs=completed,physical_frames=frames,
        actor_optimizer_steps=actor_steps,critic_optimizer_steps=critic_steps,
        seconds=time.monotonic()-began,quality_claim=False,next='Independent development evaluation required'))
    agent.writer.close();close_runtime(env)


def dynamics(args):
    import unionmimic.runtime
    import torch
    from unionmimic.models import load_world_model
    from unionmimic.dynamics_fit import DynamicsFit,EpisodeChunks
    from torch.utils.tensorboard import SummaryWriter
    gate=ReviewGate(args.manifest,args.approval,'dynamics');gate.verify_inputs();gate()
    m=gate.manifest;c=m['dynamics_pilot'];dataset=ROOT/m['dataset']
    began=time.monotonic();deadline=began+c['max_seconds']
    wm,saved,cfg=load_world_model(dataset/'config.yaml',ROOT/m['checkpoint'],args.gpu)
    torch.manual_seed(c['seed']);fit=DynamicsFit(wm,saved)
    chunks=EpisodeChunks(dataset,digest(wm.encoder),c['length'],c['seed'])
    params=fit.parameters();opt=torch.optim.Adam(params,lr=c['lr'])
    frozen=(digest(wm.encoder),digest(wm.heads['decoder']._cnn));step=0;stop='pilot_update_limit'
    original_dynamics=digest(wm.dynamics);writer=SummaryWriter(str(args.output/'tensorboard'))
    with open(args.output/'metrics.jsonl','x',buffering=1) as log:
        for update in range(1,c['updates']+1):
            if time.monotonic()>=deadline:stop='pilot_time_cap';break
            rows=chunks.sample(c['batch'],next(wm.parameters()).device);opt.zero_grad(set_to_none=True)
            loss,metrics=fit.loss(rows,c['horizon'],c['feature_drift_coefficient']);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,c['clip_norm'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite dynamics gradient')
            if time.monotonic()>=deadline:stop='pilot_time_cap';break
            gate();opt.step();step=update
            row=dict(step=step,loss=float(loss),grad_norm=float(norm),seconds=time.monotonic()-began,**metrics)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            for key,value in row.items():
                if isinstance(value,(int,float)):writer.add_scalar('dynamics/'+key,value,step)
            progress(args,phase='fit',**row)
            if step==1 or step%c['save_every']==0:
                assert frozen==(digest(wm.encoder),digest(wm.heads['decoder']._cnn))
                assert original_dynamics!=digest(wm.dynamics), 'RSSM optimizer did not change its parameters'
                save_checkpoint(args.output/('step-%04d.pt'%step),fit.checkpoint(saved,opt,step))
                print(row,flush=True)
        if step:
            assert frozen==(digest(wm.encoder),digest(wm.heads['decoder']._cnn))
            save_checkpoint(args.output/'last.pt',fit.checkpoint(saved,opt,step))
    chunks.close();writer.close()
    write_json(args.output/'completed.json',dict(status=stop,optimizer_steps=step,
        seconds=time.monotonic()-began,quality_claim=False,
        next='Recompute current-model states; validate H1/H4 and frozen-actor physical behavior before any SM update'))


def run(args):
    progress(args,phase='initialize',started_unix=time.time())
    try:
        (ppo if args.arm=='ppo' else dynamics)(args)
        report=json.loads((args.output/'completed.json').read_text())
        write_json(args.output/'status.json',dict(pid=os.getpid(),arm=args.arm,status='completed',
            updated_unix=time.time(),result=report))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(pid=os.getpid(),arm=args.arm,status='failed',
            updated_unix=time.time(),error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['ppo','dynamics'])
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args()
    args.manifest=args.manifest.resolve();args.approval=args.approval.resolve();args.output=args.output.resolve()
    # Fail before simulator allocation or output creation when not reviewed.
    gate=ReviewGate(args.manifest,args.approval,args.arm);gate.verify_inputs();gate()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.arm=='ppo':
        from unionmimic.simulation_worker import run_isolated_simulation
        run_isolated_simulation(run,args)
    else:run(args)
