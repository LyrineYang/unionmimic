"""Physical rollout + upstream PPO backward, with zero optimizer updates."""
import argparse
import subprocess
import time
from pathlib import Path
from unionmimic.runtime import make_runtime, close_runtime
import torch
from omegaconf import OmegaConf
from unionmimic.ppo import make_ppo, make_critic, sample_policy, collect_ppo_block, ppo_minibatch,update_ppo_epoch,pilot_checkpoint
from unionmimic.common import digest, file_digest, write_json,save_checkpoint
from unionmimic.simulation_worker import run_isolated_simulation


def main(args):
    args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve()
    began=time.monotonic()
    env,player,cfg,obs=make_runtime(args.envs,args.gpu,args.seed,args.checkpoint,
        overrides=['task.env.gravityCurriculum.maxAngleDeg=1.0'])
    startup=time.monotonic()-began
    device=env.device;wm=env.wm_adapter.world_model
    saved=torch.load(str(args.checkpoint),map_location='cpu')
    config=OmegaConf.to_container(cfg,resolve=True)
    agent=make_ppo(config,saved,device,args.output,args.envs,16,args.envs*16)
    critic=make_critic(wm,device)
    hashes=(digest(agent.model),digest(wm),digest(critic))
    with torch.no_grad():
        expected=player.get_action(obs,True)
        actual=sample_policy(agent,obs)['mus'].clamp(-1,1)
    parity=float((expected-actual).abs().max());assert parity<1e-6,parity
    # Warm up simulator/camera/RSSM independently of the actor loss.
    for _ in range(args.warmup):
        obs,_,_,_=player.env_step(env,player.get_action(obs,True))
    torch.cuda.synchronize(device);started=time.monotonic()
    blocks=[];invalid=0;native_memory=[]
    for _ in range(args.blocks):
        batch,obs,skipped=collect_ppo_block(env,player,agent,critic,obs)
        blocks.append(batch);invalid+=skipped
        native_memory.append(int(subprocess.check_output(['nvidia-smi','--id='+str(args.gpu),
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
    torch.cuda.synchronize(device);rollout_seconds=time.monotonic()-started
    batch={k:torch.cat([b[k] for b in blocks]) for k in blocks[0]}
    batch['advantage']=(batch['advantage']-batch['advantage'].mean())/(batch['advantage'].std(unbiased=False)+1e-8)
    ids=torch.arange(min(4096,len(batch['reward'])),device=device)
    started=time.monotonic();agent.set_train()
    agent.calc_gradients(ppo_minibatch(agent,batch,ids))
    actor_loss=float(agent.train_result[0]);critic.zero_grad(set_to_none=True)
    critic_loss=(critic(batch['critic_input'][ids]).squeeze(-1)-batch['return'][ids]).square().mean()
    critic_loss.backward()
    grads=[p.grad for p in critic.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    torch.cuda.synchronize(device);backward_seconds=time.monotonic()-started
    assert hashes==(digest(agent.model),digest(wm),digest(critic)), 'Preflight changed weights or normalization'
    assert agent.optimizer_steps==0 and len(agent.optimizer.state)==0
    epoch_report=None
    if args.roundtrip:
        critic_opt=torch.optim.Adam(critic.parameters(),lr=3e-4)
        epoch_report=update_ppo_epoch(agent,critic,critic_opt,batch,dry_run=True)
        assert agent.optimizer_steps==0 and len(critic_opt.state)==0
        assert hashes==(digest(agent.model),digest(wm),digest(critic))
        save_checkpoint(args.output/'roundtrip.pt',pilot_checkpoint(saved,agent,critic,critic_opt,0,len(batch['reward']),'unapproved-dry-run',epoch_report))
        restored=torch.load(str(args.output/'roundtrip.pt'),map_location=device)
        agent.model.load_state_dict(restored['model']);critic.load_state_dict(restored['ppo_state']['critic'])
        agent.optimizer.load_state_dict(restored['ppo_state']['actor_optimizer'])
        critic_opt.load_state_dict(restored['ppo_state']['critic_optimizer'])
        assert hashes==(digest(agent.model),digest(wm),digest(critic))
    report=dict(passed=True,mode='forward_backward_without_optimizer_step',gpu=args.gpu,
        gpu_name=torch.cuda.get_device_name(args.gpu),compute_capability=list(torch.cuda.get_device_capability(args.gpu)),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,precision='float32',envs=args.envs,
        checkpoint_sha256=file_digest(args.checkpoint),policy_parity_max_error=parity,
        optimizer_steps=0,weights_and_normalization_unchanged=True,startup_seconds=startup,
        sampled_control_transitions=args.blocks*16*args.envs,valid_transitions=len(batch['reward']),
        excluded_reset_transitions=invalid,rollout_seconds=rollout_seconds,
        rollout_transitions_per_second=args.blocks*16*args.envs/rollout_seconds,
        backward_batch=len(ids),backward_seconds=backward_seconds,actor_loss=actor_loss,
        critic_loss=float(critic_loss),actor_grad_norm=agent.last_grad_norm,
        pytorch_peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/1024**2,
        max_sampled_total_gpu_memory_mib=max(native_memory),
        bf16_supported=bool(torch.cuda.is_bf16_supported()),
        compiled_cuda_architectures=torch.cuda.get_arch_list(),
        gravity_cone_max_degrees=env.gravity_curriculum_max_angle_deg,
        full_epoch_dry_run=epoch_report,checkpoint_roundtrip_passed=args.roundtrip,
        limitations=['No optimizer update, learning-speed or time-to-target measured.',
                     'PyTorch peak excludes native PhysX/Vulkan allocations.'])
    write_json(args.output/'preflight.json',report)
    print(report,flush=True);agent.writer.close();close_runtime(env)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--envs',type=int,default=16);p.add_argument('--seed',type=int,default=51001)
    p.add_argument('--warmup',type=int,default=32);p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--roundtrip',action='store_true')
    run_isolated_simulation(main,p.parse_args())
