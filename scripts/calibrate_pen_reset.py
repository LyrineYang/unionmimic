"""Bounded contact calibration with the actual composite asset/body indices.

Two independent randomized replicas score each candidate. All clips remain in
the dataset. Candidate survival does not certify a trajectory or learned policy.
"""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import argparse,time,json,traceback
import torch
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.pen_tasks import LINK_NAMES
from unionmimic.motions import tracking_metrics
from unionmimic.common import write_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--candidates',type=int,default=96);p.add_argument('--seed',type=int,default=20261121)
    p.add_argument('--moving-and-hold',action='store_true')
    p.add_argument('--replicas',type=int,default=4)
    a=p.parse_args();root=Path(a.input).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4)
    assert a.replicas>=4 and a.replicas%2==0
    C=a.candidates;R=a.replicas if a.moving_and_hold else 2;n=13*C*R;env,cfg=make_pen_simulator(n,a.gpu,a.seed,root,fixed_phase=0)
    sources={str(x.relative_to(ROOT)):file_digest(x) for x in [Path(__file__).resolve(),ROOT/'unionmimic/pen_simulator.py',ROOT/'unionmimic/pen_tasks.py',ROOT/'unionmimic/pen_reward.py']}
    write_json(out/'run.json',dict(cli=vars(a),started_unix=start,sources=sources,config=cfg))
    env.fixed_clips=True;env.clip_ids[:]=torch.arange(n,device=env.device)//(C*R);env.task_ids[:]=1
    if a.moving_and_hold:
        moving=(torch.arange(n,device=env.device)%R)>=R//2
        env.task_ids[moving]=torch.where(env.clip_ids[moving]<4,2,3)
    q0=env.pen_tasks.initial_q.clone();p0=env.pen_tasks.initial.clone();bestq=q0.clone();bestshift=torch.zeros(13,3,device=env.device)
    original=env.pen_tasks.target;candidate_q=candidate_pose=None
    def fixed_target(task_ids,clip_ids,steps):
        result=original(task_ids,clip_ids,steps)
        assert len(task_ids)==n,'Calibration resets every candidate together'
        if a.moving_and_hold:
            offset=candidate_q-q0[clip_ids];shift=candidate_pose[:,:3]-p0[clip_ids,:3]
            result['q']=(result['q']+offset).clamp(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits)
            result['pose']=torch.cat((result['pose'][:,:3]+shift,result['pose'][:,3:]),-1)
            nxt=original(task_ids,clip_ids,steps+1)['q']
            result['qd']=((nxt+offset).clamp(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits)-result['q'])/.02
        else:
            result.update(q=candidate_q,qd=torch.zeros_like(candidate_q),pose=candidate_pose,v=torch.zeros(n,3,device=env.device),w=torch.zeros(n,3,device=env.device))
        return result
    env.pen_tasks.target=fixed_target;records=[]
    for iteration,scale in enumerate((1.,1.,.7,.5,.3,.2,.1,.05)):
        q=(bestq[:,None]+torch.randn(13,C,22,device=env.device)*.22*scale).clamp(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits)
        shift=(bestshift[:,None]+torch.randn(13,C,3,device=env.device)*.008*scale).clamp(-.025,.025)
        q[:,0]=bestq;shift[:,0]=bestshift
        candidate_q=q.reshape(-1,22).repeat_interleave(R,0);candidate_pose=p0[:,None].expand(-1,C,-1).reshape(-1,7).repeat_interleave(R,0).clone()
        candidate_pose[:,:3]+=shift.reshape(-1,3).repeat_interleave(R,0)
        env.reset_buf.fill_(1);env.step(torch.zeros(n,22,device=env.device))
        failed=env.pen_failure.clone();score=torch.zeros(n,device=env.device);peak=torch.zeros_like(score)
        for step in range(75):
            env.reset_buf.zero_();env.step(torch.zeros(n,22,device=env.device))
            goal=fixed_target(env.task_ids,env.clip_ids,env.reference_step)['pose'] if a.moving_and_hold else candidate_pose
            m=tracking_metrics(env.local_pose(),goal)
            failed|=m['failure'];score+=m['position_error_m']/.05+m['shaft_error_rad']/(torch.pi/3)
            force=env.contact_tensor.view(n,-1,3)[:,env.object_rb_handles[0]].norm(dim=-1);peak=torch.maximum(peak,force)
        bad=failed|(peak>50)
        scores=bad.float().view(13,C,R).sum(-1)*10000+score.view(13,C,R).max(-1).values
        scores+=(peak/50).clamp(max=100).view(13,C,R).max(-1).values
        choice=scores.argmin(-1);rows=torch.arange(13,device=env.device);bestq=q[rows,choice];bestshift=shift[rows,choice]
        chosen_bad=bad.view(13,C,R)[rows,choice];chosen_peak=peak.view(13,C,R)[rows,choice]
        record=dict(iteration=iteration,seconds=time.time()-start,robust_clips_passed=int((~chosen_bad.any(-1)).sum()),
            selected_bad=chosen_bad.tolist(),selected_peak_force_n=chosen_peak.tolist(),score=scores[rows,choice].tolist())
        records.append(record);write_json(out/'progress.json',records);print(json.dumps(record),flush=True)
        torch.save(dict(q=bestq.cpu(),shift=bestshift.cpu(),iteration=iteration),out/'best-candidates.pt')
    manifest=json.loads((root/'manifest.json').read_text())
    for i,info in enumerate(manifest['clips']):
        d=torch.load(root/info['file'],map_location='cpu');delta=bestq[i].cpu()-d['q'][0];k=min(50,len(d['q']))
        if a.moving_and_hold:d['q']+=delta
        else:d['q'][:k]+=delta*(1-torch.arange(k).float()/(k-1))[:,None]
        d['q']=d['q'].clamp(env.arm_hand_dof_lower_limits.cpu(),env.arm_hand_dof_upper_limits.cpu())
        d['pose'][:,:3]+=bestshift[i].cpu();d['sharpa_physics_verified']=False;d['parent_actual_links']=d.pop('actual_links',None)
        torch.save(d,out/info['file']);info['retarget_sha256']=file_digest(out/info['file'])
    manifest.update(physics_verified=False,contact_calibration=dict(body_index=env.body_index_audit,replicas=R,source_hashes=sources,
        pose_translation_m=bestshift.tolist(),results=records,scope='first 1.5s moving and hold' if a.moving_and_hold else 'initial pose only, not whole-motion verification'))
    write_json(out/'manifest.json',manifest)
if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
