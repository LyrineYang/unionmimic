"""Bounded physical acceptance, including all reference initializations, not training."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from isaacgym import gymapi
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import argparse,time,traceback,json
import torch
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.pen_tasks import PenTasks
from unionmimic.pen_reward import reward_terms
from unionmimic.hand_kinematics import HandKinematics
from unionmimic.motions import tracking_metrics,qrot,qinv
from unionmimic.common import write_json,file_digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=3);ap.add_argument('--envs',type=int,default=52)
    ap.add_argument('--out',required=True);ap.add_argument('--retarget',default=str(ROOT/'work/motions/sharpa-pen-20261104-v1'))
    ap.add_argument('--moving',action='store_true')
    ap.add_argument('--seed',type=int,default=20261107)
    ap.add_argument('--max-depenetration',type=float);ap.add_argument('--velocity-iterations',type=int)
    a=ap.parse_args();a.retarget=str(Path(a.retarget).resolve());out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4)
    source_paths=[Path(__file__).resolve(),ROOT/'unionmimic/pen_simulator.py',ROOT/'unionmimic/pen_tasks.py',ROOT/'unionmimic/pen_camera.py',ROOT/'unionmimic/pen_reward.py',ROOT/'unionmimic/pen_sampling.py',ROOT/'unionmimic/hand_kinematics.py',ROOT/'assets/reference_pen/pen_tracking220.urdf']
    write_json(out/'manifest.json',dict(started_unix=start,cli=vars(a),sources={str(p.relative_to(ROOT)):file_digest(p) for p in source_paths}))
    for p in source_paths:
        dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    numerics={}
    if a.max_depenetration is not None:numerics['max_depenetration_velocity']=a.max_depenetration
    if a.velocity_iterations is not None:numerics['num_velocity_iterations']=a.velocity_iterations
    env,cfg=make_pen_simulator(a.envs,a.gpu,a.seed,a.retarget,fixed_phase=0,physics_numerics=numerics)
    write_json(out/'config.json',cfg)
    env.fixed_clips=True;env.clip_ids[:]=torch.arange(a.envs,device=env.device)%13;env.task_ids[:]=1
    if a.moving:env.task_ids[:]=torch.where(env.clip_ids<4,2,3)
    env.reset_buf[:]=1;env.step(torch.zeros(a.envs,22,device=env.device))
    fk=HandKinematics(ROOT/'work/upstreams/WM-Craftnet/assets/urdf/right_sharpa_wave/right_sharpa_wave.urdf',env.arm_hand_dof_names,env.device)
    names=['right_'+f+'_fingertip' for f in ('thumb','index','middle','ring','pinky')]
    handles=[env.gym.find_actor_rigid_body_handle(env.envs[0],env.arm_hands[0],x) for x in names]
    base=env.hand_start_states;bq=base[:,[6,3,4,5]]
    actual=qrot(qinv(bq)[:,None].expand(-1,5,-1),env.rigid_body_states[:,handles,:3]-base[:,None,:3])
    fkerr=float((actual-fk.forward(env.arm_hand_dof_pos,names)).norm(dim=-1).max())
    rows=[];failed=env.pen_failure.clone();first_fail=torch.where(failed,torch.zeros(a.envs,device=env.device),torch.full((a.envs,),-1,device=env.device))
    peak_object_force=torch.zeros(a.envs,device=env.device)
    initial=env.pen_tasks.initial[env.clip_ids].clone()
    for step in range(50):
        # Observe the first initialization for one physical second. Do not respawn
        # failed objects and then count the respawn as a successful hold.
        env.reset_buf.zero_();env.step(torch.zeros(a.envs,22,device=env.device))
        goal=env.pen_tasks.target(env.task_ids,env.clip_ids,env.reference_step)['pose'] if a.moving else initial
        m=tracking_metrics(env.local_pose(),goal)
        first_fail[(first_fail<0)&m['failure']]=step+1;failed|=m['failure']
        force=env.contact_tensor.view(a.envs,-1,3).norm(dim=-1)
        peak_object_force=torch.maximum(peak_object_force,force[:,env.object_rb_handles].max(-1).values)
        if step in (0,4,24,49):
            rows.append(dict(seconds=(step+1)*.02,position_m=m['position_error_m'].tolist(),angle_rad=m['shaft_error_rad'].tolist(),
                finger_contacts=(env.contact_tensor.view(a.envs,-1,3).norm(dim=-1)[:,env.sensor_handle_indices]>1).sum(-1).tolist()))
        assert torch.isfinite(env.policy_frame).all() and torch.isfinite(env.root_state_tensor).all()
    # Verify the shared four-task controller/reward/termination interface, separately
    # from the grasp gate. This is NOT evidence of actor/critic learning.
    env.fixed_clips=False;env.task_ids[:]=torch.arange(a.envs,device=env.device)%4;env.reset_buf[:]=1
    command_error=target_error=reward_error=0.;valid_count=0
    for step in range(40):
        valid=~env.reset_buf.bool();old_action=env.last_actions.clone()
        ref=env.pen_tasks.target(env.task_ids,env.clip_ids,env.reference_step)['q']
        env.step(torch.randn(a.envs,22,device=env.device)*.1)
        if valid.any():
            action=.5*(env.study_command*.5).clamp(-.6,.6)+.5*old_action
            target=(ref+action).clamp(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits)
            command_error=max(command_error,float((action-env.last_actions)[valid].abs().max()))
            target_error=max(target_error,float((target-env.prev_targets)[valid].abs().max()))
            terms,failure,timeout,_=reward_terms(env.pen_tasks,env.local_pose(),env.arm_hand_dof_pos,env.torques,env.current_charge,
                env.raw_previous,env.transition_previous,env.transition_older,env.task_ids,env.clip_ids,env.reference_step,
                env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits,env.arm_hand_dof_effort_limits)
            reward=sum(terms.values())*.02
            reward_error=max(reward_error,float((reward-env.rew_buf)[valid].abs().max()))
            assert torch.equal((failure|timeout)[valid],env.reset_buf[valid].bool())
            valid_count+=int(valid.sum())
        assert env.policy_frame.shape[-1]==174 and torch.isfinite(env.policy_frame).all()
    interface_passed=valid_count>0 and max(command_error,target_error)<1e-6 and reward_error<1e-6
    acceptable=(~failed)&(peak_object_force<=50)
    passed=bool(fkerr<1e-4 and acceptable.float().mean()>=.9 and interface_passed)
    result=dict(passed=passed,kind='reference tracking with zero residual' if a.moving else 'initial-grasp engineering acceptance; not learned policy',envs=a.envs,
        body_index_audit=env.body_index_audit,
        control_dt=env.dt*env.control_freq_inv,ema=env.act_moving_average,relative_scale=env.relative_scale,
        fk_max_error_m=fkerr,one_second_survival=float((~failed).float().mean()),first_failure_steps=first_fail.tolist(),
        peak_object_force_n=peak_object_force.tolist(),acceptable_grasp_fraction=float(acceptable.float().mean()),
        four_task_interface=dict(passed=interface_passed,valid_transitions=valid_count,command_error=command_error,target_error=target_error,reward_error=reward_error),
        all_13_clips_covered=True,gate=dict(fk_max_error_m=1e-4,acceptable_grasp_fraction_min=.9,peak_object_force_max_n=50),rows=rows,
        elapsed=time.time()-start,configuration=cfg,retarget_manifest_sha256=file_digest(Path(a.retarget)/'manifest.json'))
    write_json(out/'result.json',result);torch.save(dict(q=env.arm_hand_dof_pos.cpu(),pose=env.local_pose().cpu(),clip_ids=env.clip_ids.cpu()),out/'last-state.pt')
    print(json.dumps({k:v for k,v in result.items() if k not in ('configuration','rows','first_failure_steps')}),flush=True)

if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
