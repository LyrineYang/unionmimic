"""Read-only policy evaluation with per-episode reward decomposition."""
import argparse
from pathlib import Path
from unionmimic.runtime import make_runtime, close_runtime
import torch
from unionmimic.collect import terminal_flags
from unionmimic.common import write_json, file_digest,save_checkpoint,digest
from unionmimic.simulation_worker import run_isolated_simulation
import isaacgymenvs.tasks.wm_craftnet_rotation as task


def evaluate(args):
    args.checkpoint=args.checkpoint.resolve()
    args.output=args.output.resolve()
    original=task.compute_hand_reward_finger
    captured={}
    def observe(*a,**kw):
        result=original(*a,**kw)
        captured['result']=result
        return result
    task.compute_hand_reward_finger=observe
    env,player,cfg,obs=make_runtime(args.envs,args.gpu,args.seed,args.checkpoint,
        overrides=['task.env.gravityCurriculum.maxAngleDeg=1.0'])
    hashes=(digest(player.model),digest(env.wm_adapter.world_model))
    if args.save_starts:
        save_checkpoint(args.output.with_suffix('.starts.pt'),dict(
            seed=args.seed,root=env.root_state_tensor.cpu().clone(),dof=env.dof_state.cpu().clone(),
            targets=env.prev_targets.cpu().clone(),last_actions=env.last_actions.cpu().clone(),
            observation={k:v.cpu().clone() for k,v in obs.items() if torch.is_tensor(v)},
            note='Exposed initial tensors only; no guarantee of hidden PhysX solver-state replay'))
    names=['spin','linear_velocity','contact','finger_distance','torque','work','action',
           'control_error','hand_pose','object_drift','off_axis','no_spin']
    totals={k:torch.zeros(args.envs,device=env.device) for k in names+['extra_contact_and_drop','total','rotation_radians']}
    live=torch.ones(args.envs,device=env.device,dtype=torch.bool)
    lengths=torch.zeros_like(live,dtype=torch.long);failures=torch.zeros_like(live)
    for step in range(env.max_episode_length+2):
        action=player.get_action(obs,True)
        obs,_,_,_=player.env_step(env,action)
        core=captured['result']
        for i,k in enumerate(names):totals[k]+=core[6+i]*live
        subtotal=sum(core[6+i] for i in range(len(names)))
        totals['extra_contact_and_drop']+=(env.rew_buf-subtotal)*live
        totals['total']+=env.rew_buf*live
        totals['rotation_radians']+=env.last_spin_delta_axis/20*live
        lengths+=live.long()
        ending=live & env.reset_buf.bool()
        failures|=ending & terminal_flags(env)['failure']
        live&=~ending
        if (step+1)%100==0:print('reward probe steps',step+1,flush=True)
        if not live.any():break
    assert not live.any()
    component_sum=sum(totals[k] for k in names+['extra_contact_and_drop'])
    error=float((component_sum-totals['total']).abs().max())
    assert error<.02,error
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=file_digest(args.checkpoint),seed=args.seed,
                gpu=args.gpu,episodes=args.envs,lengths=lengths.tolist(),failures=int(failures.sum()),
                mean_return=float(totals['total'].mean()),returns=totals['total'].tolist(),
                mean_components={k:float(v.mean()) for k,v in totals.items()},
                episode_components={k:v.tolist() for k,v in totals.items()},
                component_sum_max_error=error,action_conservative_scale=env.action_conservative_scale,
                control_dt=float(cfg['sim']['dt']) if 'sim' in cfg else float(cfg['task']['sim']['dt']),
                control_frequency_inv=env.control_freq_inv)
    assert hashes==(digest(player.model),digest(env.wm_adapter.world_model))
    report['weights_unchanged']=True
    report['gravity_cone_max_degrees']=env.gravity_curriculum_max_angle_deg
    write_json(args.output,report)
    print(report,flush=True)
    close_runtime(env)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--envs',type=int,default=8)
    p.add_argument('--save-starts',action='store_true')
    run_isolated_simulation(evaluate,p.parse_args())
