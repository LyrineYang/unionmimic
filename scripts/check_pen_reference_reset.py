"""All-clip, whole-duration reference-state reset and first-step contact check."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import argparse,time,json,traceback
import torch
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.motions import qrot,qinv
from unionmimic.common import write_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--retarget',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261171);p.add_argument('--midpoints',action='store_true')
    p.add_argument('--random-phases',action='store_true')
    p.add_argument('--max-depenetration',type=float);p.add_argument('--velocity-iterations',type=int)
    a=p.parse_args();root=Path(a.retarget).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4)
    numerics={}
    if a.max_depenetration is not None:numerics['max_depenetration_velocity']=a.max_depenetration
    if a.velocity_iterations is not None:numerics['num_velocity_iterations']=a.velocity_iterations
    n=13*32;env,cfg=make_pen_simulator(n,a.gpu,a.seed,root,fixed_phase=0,physics_numerics=numerics)
    env.fixed_clips=True;env.clip_ids[:]=torch.arange(n,device=env.device)//32
    env.task_ids[:]=torch.where(env.clip_ids<4,2,3)
    phase=(torch.arange(n,device=env.device)%32)*(env.pen_tasks.lengths[env.clip_ids]-2)//31
    if a.midpoints:phase=(((torch.arange(n,device=env.device)%32).float()+.5)*(env.pen_tasks.lengths[env.clip_ids]-2)/32).long()
    assert not (a.midpoints and a.random_phases)
    if a.random_phases:
        generator=torch.Generator(device=env.device).manual_seed(a.seed+100000)
        phase=(((torch.arange(n,device=env.device)%32).float()+torch.rand(n,device=env.device,generator=generator))*(env.pen_tasks.lengths[env.clip_ids]-2)/32).long()
    # 32 positions per clip across its entire duration, not just fitted first frames.
    env.fixed_phase=phase;ids=torch.arange(n,device=env.device)
    env.reset_idx(ids,ids[:0]);reference=env.pen_tasks.target(env.task_ids,env.clip_ids,phase)
    obj=env.root_state_tensor[env.object_indices];base=env.hand_start_states;bq=base[:,[6,3,4,5]]
    errors=dict(q=float((env.arm_hand_dof_pos-reference['q']).abs().max()),qd=float((env.arm_hand_dof_vel-reference['qd']).abs().max()),
        pose=float((env.local_pose()-reference['pose']).abs().max()),
        v=float((qrot(qinv(bq),obj[:,7:10])-reference['v']).abs().max()),w=float((qrot(qinv(bq),obj[:,10:13])-reference['w']).abs().max()))
    assert max(errors.values())<1e-5,errors
    assert not env.pose_sensor.ready.any() and not env.current_charge.any() and not env.raw_previous.any()
    env.reset_buf.zero_();env.step(torch.zeros(n,22,device=env.device))
    force=env.contact_tensor.view(n,-1,3)[:,env.object_rb_handles[0]].norm(dim=-1)
    acceptable=(~env.pen_failure)&(force<=50)
    result=dict(reset_contract_passed=True,physical_first_step_passed=bool(acceptable.float().mean()>=.9),
        all_clips=13,phases_per_clip=32,errors=errors,first_step_acceptable=int(acceptable.sum()),n=n,
        per_clip_acceptable=acceptable.reshape(13,32).sum(-1).tolist(),force_n=force.tolist(),failure=env.pen_failure.tolist(),
        phases=phase.tolist(),full_motion_policy_success=False,seconds=time.time()-start,
        retarget_manifest_sha256=file_digest(root/'manifest.json'),seed=a.seed,midpoints=a.midpoints,random_phases=a.random_phases,physics_numerics=cfg['sim']['physx'])
    write_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('force_n','failure','phases')}),flush=True)
if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
