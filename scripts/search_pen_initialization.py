"""Bounded simulator search for per-clip reset grasps; no actor training.

All candidates keep the original retargeted object pose and physical limits.
The independent validation seed is never used to select a grasp.
"""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from isaacgym import gymtorch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import argparse,json,time,traceback
import torch
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.motions import tracking_metrics
from unionmimic.common import write_json,file_digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--input',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--candidates',type=int,default=64);a=ap.parse_args()
    root=Path(a.input).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.time();torch.set_num_threads(4);N=13*a.candidates
    sources=[Path(__file__).resolve(),ROOT/'unionmimic/pen_simulator.py',ROOT/'unionmimic/pen_tasks.py',ROOT/'unionmimic/pen_camera.py']
    write_json(out/'search-manifest.json',dict(started_unix=start,cli=vars(a),sources={str(p.relative_to(ROOT)):file_digest(p) for p in sources}))
    for p in sources:
        dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    env,cfg=make_pen_simulator(N,a.gpu,20261108,root);env.fixed_clips=True
    env.clip_ids[:]=torch.arange(N,device=env.device)//a.candidates;env.task_ids[:]=1
    ids=torch.arange(N,device=env.device);empty=ids[:0];hi=env.hand_indices.to(torch.int32)
    baseline=env.pen_tasks.initial_q.clone();best=baseline.clone();summaries=[]
    for round_id,scale in enumerate((.12,.06,.025)):
        q=best[env.clip_ids]+torch.randn(N,22,device=env.device)*scale
        q[::a.candidates]=best
        q=q.clamp(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits)
        env.reset_idx(ids,empty)
        env.arm_hand_dof_pos[:]=q;env.arm_hand_dof_vel.zero_();env.prev_targets[:]=q;env.cur_targets[:]=q
        env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(hi),N)
        env.gym.set_dof_position_target_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.prev_targets),gymtorch.unwrap_tensor(hi),N)
        failed=torch.zeros(N,dtype=torch.bool,device=env.device);score=torch.zeros(N,device=env.device)
        peak=torch.zeros(N,device=env.device);goal=env.pen_tasks.initial[env.clip_ids]
        for step in range(50):
            env.reset_buf.zero_();env.step(torch.zeros(N,22,device=env.device))
            m=tracking_metrics(env.local_pose(),goal);failed|=m['failure']
            score+=m['position_error_m']/.05+m['shaft_error_rad']/(torch.pi/3)
            contact=env.contact_tensor.view(N,-1,3).norm(dim=-1)
            peak=torch.maximum(peak,contact.max(-1).values)
        # Failures and large impact impulses cannot win over a viable candidate.
        score+=failed.float()*10000+(peak>50).float()*10000
        selection=score.view(13,-1).argmin(1)+torch.arange(13,device=env.device)*a.candidates
        best=q[selection].clone()
        rows=dict(round=round_id,scale_rad=scale,surviving_candidates=int((~failed).sum()),
            selected_survival=(~failed[selection]).tolist(),selected_peak_force_n=peak[selection].tolist(),
            selected_score=score[selection].tolist(),seconds=time.time()-start)
        summaries.append(rows);write_json(out/'search-progress.json',summaries);print(json.dumps(rows),flush=True)
    manifest=json.loads((root/'manifest.json').read_text())
    for i,c in enumerate(manifest['clips']):
        d=torch.load(root/c['file'],map_location='cpu');old=d['q'][0].clone();delta=best[i].cpu()-old
        n=min(50,len(d['q']));weight=(1-torch.arange(n).float()/max(n-1,1))[:,None]
        d['q'][:n]+=weight*delta;d['q']=d['q'].clamp(env.arm_hand_dof_lower_limits.cpu(),env.arm_hand_dof_upper_limits.cpu())
        d['reset_search']=True;d['sharpa_physics_verified']=False;torch.save(d,out/c['file'])
    manifest.update(physics_verified=False,initialization_search=dict(seed=20261108,candidates_per_clip=a.candidates,rounds=3,
        object_target_changed=False,first_second_joint_reference_blend=True,parent=str(root),seconds=time.time()-start,results=summaries))
    write_json(out/'manifest.json',manifest)

if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
