"""Bounded four-lane offline calibration of reference-state reset anchors.

Does not train a policy. Preserves timestamps, object rotation and hardware
limits. A wrist-local object-position correction is bounded to 5 mm and is
reported explicitly; this is a Sharpa retarget, not unchanged Wuji motion.
"""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import argparse,time,json,traceback
import torch
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.motions import tracking_metrics
from unionmimic.common import write_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--candidates',type=int,default=32);p.add_argument('--anchors',type=int,default=32)
    p.add_argument('--lane',type=int);p.add_argument('--lanes',type=int,default=4)
    a=p.parse_args();root=Path(a.input).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    lane=a.gpu if a.lane is None else a.lane;assert 0<=lane<a.lanes
    start=time.time();torch.set_num_threads(4);C=a.candidates;R=2;nodes=torch.arange(13*a.anchors,device='cuda:%d'%a.gpu)[lane::a.lanes];N=len(nodes);n=N*C*R
    env,cfg=make_pen_simulator(n,a.gpu,20261180+a.gpu,root,fixed_phase=0)
    clips=nodes//a.anchors;phase=(nodes%a.anchors)*(env.pen_tasks.lengths[clips]-2)//(a.anchors-1)
    env.fixed_clips=True;env.clip_ids[:]=clips.repeat_interleave(C*R);env.task_ids[:]=torch.where(env.clip_ids<4,2,3)
    env.fixed_phase=phase.repeat_interleave(C*R)
    original=env.pen_tasks.target;q_offset=pose_offset=None
    def targets(task_ids,clip_ids,steps):
        ref=original(task_ids,clip_ids,steps)
        ref['q']=(ref['q']+q_offset).clamp(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits)
        ref['qd']=((original(task_ids,clip_ids,steps+1)['q']+q_offset).clamp(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits)-ref['q'])/.02
        ref['pose']=torch.cat((ref['pose'][:,:3]+pose_offset,ref['pose'][:,3:]),-1)
        return ref
    env.pen_tasks.target=targets
    bestq=torch.zeros(N,22,device=env.device);bestp=torch.zeros(N,3,device=env.device);history=[]
    for iteration,scale in enumerate((1.,.8,.6,.5,.4,.3,.2,.15,.1,.07,.05,.03)):
        q=(bestq[:,None]+torch.randn(N,C,22,device=env.device)*.18*scale).clamp(-.6,.6)
        pos=(bestp[:,None]+torch.randn(N,C,3,device=env.device)*.002*scale).clamp(-.005,.005)
        q[:,0]=bestq;pos[:,0]=bestp
        # Preserve the independently verified first-frame calibration.
        q[nodes%a.anchors==0]=0;pos[nodes%a.anchors==0]=0
        q_offset=q.reshape(-1,22).repeat_interleave(R,0);pose_offset=pos.reshape(-1,3).repeat_interleave(R,0)
        env.reset_buf.fill_(1);failed=torch.zeros(n,dtype=torch.bool,device=env.device);peak=torch.zeros(n,device=env.device);score=torch.zeros_like(peak)
        for step in range(10):
            if step:env.reset_buf.zero_()
            env.step(torch.zeros(n,22,device=env.device));ref=targets(env.task_ids,env.clip_ids,env.reference_step)
            m=tracking_metrics(env.local_pose(),ref['pose']);failed|=m['failure']
            force=env.contact_tensor.view(n,-1,3)[:,env.object_rb_handles[0]].norm(dim=-1);peak=torch.maximum(peak,force)
            score+=m['position_error_m']/.05+m['shaft_error_rad']/(torch.pi/3)
        bad=failed|(peak>50)
        # Count all failed replicas; a worst-case Boolean alone cannot distinguish
        # candidates with one failure from candidates failing in every replica.
        objective=bad.float().reshape(N,C,R).sum(-1)*10000
        objective+=score.reshape(N,C,R).max(-1).values+(peak/50).clamp(max=100).reshape(N,C,R).max(-1).values
        objective+=.1*q.square().sum(-1)+100*pos.square().sum(-1)
        choice=objective.argmin(-1);rows=torch.arange(N,device=env.device);bestq=q[rows,choice];bestp=pos[rows,choice]
        passed=(~bad.reshape(N,C,R)[rows,choice].any(-1));history.append(dict(iteration=iteration,passed=int(passed.sum()),nodes=N,seconds=time.time()-start))
        write_json(out/'progress.json',history);print(json.dumps(history[-1]),flush=True)
    torch.save(dict(nodes=nodes.cpu(),clips=clips.cpu(),phases=phase.cpu(),q_offset=bestq.cpu(),position_offset=bestp.cpu()),out/'anchors.pt')
    write_json(out/'result.json',dict(completed=True,source_manifest_sha256=file_digest(root/'manifest.json'),history=history,
        initialized_state_only=True,physical_acceptance_pending=True,position_bound_m=.005,anchors_per_clip=a.anchors,lanes=a.lanes,lane=lane,source_sha256=file_digest(ROOT/'scripts/calibrate_pen_phases.py')))
if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
