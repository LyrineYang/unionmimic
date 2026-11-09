"""Offline reset calibration against physics; old simulator is isolated in a source snapshot."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from isaacgym import gymtorch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/pen-prototype-20261104-v1'))
import argparse,json,time,traceback
import torch
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.motions import qrot,qmul,tracking_metrics
from unionmimic.common import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--candidates',type=int,default=128);ap.add_argument('--seed',type=int,default=20261112);ap.add_argument('--position-range',type=float,default=.015)
    a=ap.parse_args();root=Path(a.input).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);start=time.time();n=13*a.candidates;env,cfg=make_pen_simulator(n,a.gpu,a.seed,root)
    env.fixed_clips=True;env.clip_ids[:]=torch.arange(n,device=env.device)//a.candidates;env.task_ids[:]=1
    ids=torch.arange(n,device=env.device);empty=ids[:0];hi=env.hand_indices.to(torch.int32);oi=env.object_indices.to(torch.int32)
    q0=env.pen_tasks.initial_q.clone();pose0=env.pen_tasks.initial.clone();bestq=q0.clone();bestshift=torch.zeros(13,3,device=env.device)
    bestscore=torch.full((13,),float('inf'),device=env.device);records=[]
    for rid,scale in enumerate((1.,.6,.3,.15,.07)):
        q=(bestq[env.clip_ids]+torch.randn(n,22,device=env.device)*.15*scale).clamp(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits)
        shift=(bestshift[env.clip_ids]+torch.randn(n,3,device=env.device)*a.position_range*scale).clamp(-a.position_range,a.position_range)
        q[::a.candidates]=bestq;shift[::a.candidates]=bestshift
        goal=pose0[env.clip_ids].clone();goal[:,:3]+=shift
        env.reset_idx(ids,empty);env.arm_hand_dof_pos[:]=q;env.arm_hand_dof_vel.zero_();env.prev_targets[:]=q;env.cur_targets[:]=q
        base=env.hand_start_states;bq=base[:,[6,3,4,5]];worldp=qrot(bq,goal[:,:3])+base[:,:3]
        env.root_state_tensor[env.object_indices,:3]=worldp;env.root_state_tensor[env.object_indices,7:]=0
        env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor),gymtorch.unwrap_tensor(oi),n)
        env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(hi),n)
        env.gym.set_dof_position_target_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.prev_targets),gymtorch.unwrap_tensor(hi),n)
        failed=torch.zeros(n,dtype=torch.bool,device=env.device);score=torch.zeros(n,device=env.device);peak=torch.zeros(n,device=env.device)
        for step in range(75):
            env.reset_buf.zero_();env.step(torch.zeros(n,22,device=env.device));m=tracking_metrics(env.local_pose(),goal)
            failed|=m['failure'];score+=m['position_error_m']/.05+m['shaft_error_rad']/(torch.pi/3)
            force=env.contact_tensor.view(n,-1,3).norm(dim=-1)[:,env.object_rb_handles].max(-1).values;peak=torch.maximum(peak,force)
        score+=failed.float()*10000+(peak>50).float()*10000
        chosen=score.view(13,-1).argmin(1)+torch.arange(13,device=env.device)*a.candidates
        bestq=q[chosen].clone();bestshift=shift[chosen].clone();bestscore=score[chosen].clone()
        record=dict(round=rid,selected_survival=(~failed[chosen]).tolist(),selected_force_n=peak[chosen].tolist(),selected_score=bestscore.tolist(),seconds=time.time()-start)
        records.append(record);write_json(out/'progress.json',records);print(json.dumps(record),flush=True)
    manifest=json.loads((root/'manifest.json').read_text())
    for i,info in enumerate(manifest['clips']):
        d=torch.load(root/info['file'],map_location='cpu');delta=bestq[i].cpu()-d['q'][0];k=min(50,len(d['q']))
        d['q'][:k]+=delta*(1-torch.arange(k).float()/(k-1))[:,None];d['q']=d['q'].clamp(env.arm_hand_dof_lower_limits.cpu(),env.arm_hand_dof_upper_limits.cpu())
        d['pose'][:,:3]+=bestshift[i].cpu();d['sharpa_physics_verified']=False
        d['parent_actual_links']=d.pop('actual_links',None);torch.save(d,out/info['file'])
    manifest.update(physics_verified=False,reset_calibration=dict(seed=a.seed,pose_translation_m=bestshift.cpu().tolist(),
        original_temporal_speed_preserved=True,reference_object_trajectory_translated=True,seconds=time.time()-start,results=records))
    write_json(out/'manifest.json',manifest)
if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
