"""Resumable PPO segment. Simulator state is deliberately recollected on restart."""
import argparse,json,os,time
from pathlib import Path
from unionmimic.common import ROOT,digest,save_checkpoint,write_json


def restore_ppo(saved,agent,critic,critic_opt):
    import torch
    state=saved.get('ppo_state')
    if state is None:return 0,0,0
    critic.load_state_dict(state['critic'],strict=True)
    agent.optimizer.load_state_dict(state['actor_optimizer'])
    critic_opt.load_state_dict(state['critic_optimizer'])
    agent.last_lr=state['actor_lr'];agent.update_lr(agent.last_lr)
    agent.optimizer_steps=state.get('actor_steps',saved.get('ppo_report',{}).get('actor_steps',0))
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'],agent.ppo_device)
    return state['epoch'],state['physical_frames'],state.get('critic_steps',0)


def train(a):
    from unionmimic.runtime import make_runtime,close_runtime
    import torch
    from omegaconf import OmegaConf
    from unionmimic.ppo import make_ppo,make_critic,collect_ppo_block,update_ppo_epoch,pilot_checkpoint
    from unionmimic.review_gate import ReviewGate
    gate=ReviewGate(a.manifest,a.approval,'ppo');gate.verify_inputs();gate()
    c=gate.manifest['ppo'];deadline=time.monotonic()+max(0,a.deadline-time.time())
    if time.monotonic()>=deadline:raise TimeoutError('Original training deadline has passed')
    env,player,cfg,obs=make_runtime(c['envs'],a.gpu,a.seed,a.checkpoint,
        overrides=['task.env.gravityCurriculum.maxAngleDeg=1.0'])
    saved=torch.load(str(a.checkpoint),map_location='cpu');wm=env.wm_adapter.world_model
    wm.requires_grad_(False)
    agent=make_ppo(OmegaConf.to_container(cfg,resolve=True),saved,env.device,a.output,
                  c['envs'],16,4096,c['actor_lr'],False,gate)
    critic=make_critic(wm,env.device);opt=torch.optim.Adam(critic.parameters(),lr=3e-4)
    previous,frames,critic_steps=restore_ppo(saved,agent,critic,opt)
    fixed=(digest(wm),digest(agent.model.running_mean_std),digest(agent.model.value_mean_std))
    initial_actor=digest(agent.model);began=time.monotonic();last=None
    with open(a.output/'metrics.jsonl','x',buffering=1) as log:
        for epoch in range(previous+1,previous+a.epochs+1):
            if time.monotonic()>=deadline:break
            write_json(a.output/'status.json',dict(status='running',phase='collect',epoch=epoch,physical_frames=frames,
                updated_unix=time.time(),pid=os.getpid()))
            blocks=[];n=0;start=time.monotonic()
            while n<c['batch_frames'] and time.monotonic()<deadline:
                b,obs,invalid=collect_ppo_block(env,player,agent,critic,obs,16)
                blocks.append(b);n+=len(b['reward']);frames+=16*c['envs']
            if n<c['batch_frames']:break
            data={k:torch.cat([b[k] for b in blocks]) for k in blocks[0]}
            sampling=time.monotonic()-start;agent.epoch_num=epoch
            result=update_ppo_epoch(agent,critic,opt,data,4096,4,deadline,gate,False)
            if result is None:break
            critic_steps+=result['critic_steps']
            assert fixed==(digest(wm),digest(agent.model.running_mean_std),digest(agent.model.value_mean_std))
            row=dict(epoch=epoch,physical_frames=frames,actor_steps=agent.optimizer_steps,
                     critic_steps_total=critic_steps,**result,sample_seconds=sampling,
                     seconds=time.monotonic()-began,frozen_modules_unchanged=True,
                     actor_changed=digest(agent.model)!=initial_actor)
            assert row['actor_changed']
            log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
            for k,v in row.items():
                if isinstance(v,(int,float)):agent.writer.add_scalar('ppo/'+k,v,frames)
            agent.writer.flush()
            bundle=pilot_checkpoint(saved,agent,critic,opt,epoch,frames,gate.sha,row)
            bundle['ppo_state'].update(actor_steps=agent.optimizer_steps,critic_steps=critic_steps)
            save_checkpoint(a.output/'last.pt',bundle);last=row
            write_json(a.output/'status.json',dict(status='running',phase='update',updated_unix=time.time(),**row))
    if last is None:raise TimeoutError('No PPO update completed within the remaining budget')
    write_json(a.output/'completed.json',dict(**last,checkpoint=str(a.output/'last.pt'),
        restored_epoch=previous,simulator_resume='fresh rollout; no exact PhysX state claim'))
    agent.writer.close();close_runtime(env)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','output','manifest','approval'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    for name in ('checkpoint','output','manifest','approval'):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    from unionmimic.simulation_worker import run_isolated_simulation
    run_isolated_simulation(train,a)
