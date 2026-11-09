"""Time-based, hash-checked Wuji reference targets, independent of the robot policy.

Source data and conventions: Dex-Supermimic 2becb738, commands._read_trajectory.
Quaternions here are wxyz. The 20 source joints are deliberately NOT converted
to 22 Sharpa joints; morphology/contact retargeting is a separate acceptance.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch

def qmul(a,b):
    w=a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdim=True)
    v=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+torch.cross(a[...,1:],b[...,1:],dim=-1)
    return torch.cat((w,v),-1)

def qinv(q):return torch.cat((q[...,:1],-q[...,1:]),-1)

def qrot(q,v):
    xyz=q[...,1:];uv=torch.cross(xyz,v,dim=-1)
    return v+2*(q[...,:1]*uv+torch.cross(xyz,uv,dim=-1))

def qslerp(a,b,u):
    dot=(a*b).sum(-1,keepdim=True);b=torch.where(dot<0,-b,b);dot=dot.abs().clamp(max=1.)
    theta=torch.acos(dot.clamp(max=1.-1e-7));den=torch.sin(theta).clamp_min(1e-7)
    sphere=(torch.sin((1-u)*theta)*a+torch.sin(u*theta)*b)/den
    linear=(1-u)*a+u*b
    return torch.nn.functional.normalize(torch.where(dot>.9995,linear,sphere),dim=-1)

def wrist_local(wrist,obj,links):
    wq=torch.nn.functional.normalize(wrist[:,3:],dim=-1);oq=torch.nn.functional.normalize(obj[:,3:],dim=-1)
    iq=qinv(wq)
    local=torch.cat((qrot(iq,obj[:,:3]-wrist[:,:3]),qmul(iq,oq)),-1)
    joints=qrot(iq[:,None].expand(-1,links.shape[1],-1),links-wrist[:,None,:3])
    return local,joints

class MotionLibrary:
    GROUPS=('02_continuous_single_axis','03_continuous_multi_axis')
    def __init__(self,root,device='cpu'):
        root=Path(root).resolve();self.clips=[];self.device=device
        for group in self.GROUPS:
            index=json.loads((root/group/'index.json').read_text());dt=float(index['frame_dt'])
            if abs(dt-.02)>1e-9:raise ValueError('Expected original 50 Hz reference')
            for entry in index['clips']:
                path=(root/group/entry['file']).resolve()
                if path.parent!=root/group:raise ValueError('Reference path leaves its group')
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                if sha!=entry['sha256']:raise ValueError('Reference hash mismatch: '+str(path))
                with np.load(path,allow_pickle=False) as archive:
                    data={k:torch.as_tensor(archive[k].copy(),device=device,dtype=torch.float32)
                          for k in ('object_pose_w','wrist_pose_w','link_pos_w','reference_qpos')}
                n=len(data['reference_qpos'])
                for k,shape in [('object_pose_w',(n,7)),('wrist_pose_w',(n,7)),('link_pos_w',(n,21,3)),('reference_qpos',(n,20))]:
                    if tuple(data[k].shape)!=shape or not torch.isfinite(data[k]).all():raise ValueError('Invalid '+k)
                if n<2 or any((data[k][:,3:].norm(dim=-1)<1e-6).any() for k in ('object_pose_w','wrist_pose_w')):
                    raise ValueError('Invalid quaternion or clip length')
                pose,links=wrist_local(data['wrist_pose_w'],data['object_pose_w'],data['link_pos_w'])
                self.clips.append(dict(id=group+'/'+entry['clip_id'],group=group,dt=dt,duration=(n-1)*dt,
                    frames=n,sha256=sha,pose=pose,links=links,source_joints=data['reference_qpos']))
        self.ends=torch.tensor([x['duration'] for x in self.clips],device=device)

    def sample(self,clip_ids,seconds):
        """Known target trajectories only. Query time is seconds, never an implicit frame counter."""
        if clip_ids.shape!=seconds.shape or (clip_ids<0).any() or (clip_ids>=len(self.clips)).any():raise ValueError('Invalid clip query')
        if not torch.isfinite(seconds).all() or (seconds<0).any():raise ValueError('Invalid reference time')
        poses=torch.empty(*seconds.shape,7,device=seconds.device);links=torch.empty(*seconds.shape,21,3,device=seconds.device)
        source=torch.empty(*seconds.shape,20,device=seconds.device)
        for i,c in enumerate(self.clips):
            mask=clip_ids==i
            if not mask.any():continue
            t=(seconds[mask]/c['dt']).clamp(0,c['frames']-1);lo=t.floor().long();hi=(lo+1).clamp(max=c['frames']-1);u=(t-lo)[:,None]
            a,b=c['pose'][lo],c['pose'][hi]
            poses[mask]=torch.cat(((1-u)*a[:,:3]+u*b[:,:3],qslerp(a[:,3:],b[:,3:],u)),-1)
            links[mask]=(1-u[:,:,None])*c['links'][lo]+u[:,:,None]*c['links'][hi]
            source[mask]=(1-u)*c['source_joints'][lo]+u*c['source_joints'][hi]
        return dict(pose=poses,links=links,source_joints=source,ended=seconds>=self.ends[clip_ids])

    def validate_environment(self,*,control_dt,has_sharpa_retarget,has_pen_contact_initialization):
        if abs(control_dt-.02)>2e-5:raise ValueError('Reference training requires verified 50 Hz control; no fivefold slow playback')
        if not has_sharpa_retarget or not has_pen_contact_initialization:
            raise ValueError('Sharpa morphology/contact initialization has not been verified')

class ReferenceClock:
    """Same time rule for real collection and imagination, restored from replay."""
    def __init__(self,library,clip_ids,seconds,control_dt):
        self.library=library;self.clip_ids=clip_ids.clone();self.seconds=seconds.clone();self.dt=float(control_dt)
        if not 0<self.dt<=.1:raise ValueError('Unsupported control period')
    def target(self,offset_steps=0):return self.library.sample(self.clip_ids,self.seconds+offset_steps*self.dt)
    def advance(self):self.seconds+=self.dt;return self.target()
    def reset(self,ids,clip_ids,start_seconds):
        self.clip_ids[ids]=clip_ids;self.seconds[ids]=start_seconds
    def snapshot(self):return dict(clip_ids=self.clip_ids.clone(),seconds=self.seconds.clone(),control_dt=self.dt)

def tracking_metrics(actual_pose,target_pose):
    """Wuji train/eval boundaries: 5 cm position and 60 degree directed shaft error."""
    # The released pen's shaft is local +X, not the usual cylinder +Z axis.
    axis=torch.zeros_like(actual_pose[...,:3]);axis[...,0]=1
    ap=qrot(actual_pose[...,3:],axis);tp=qrot(target_pose[...,3:],axis)
    angle=torch.acos((ap*tp).sum(-1).clamp(-1,1));pos=(actual_pose[...,:3]-target_pose[...,:3]).norm(dim=-1)
    return dict(position_error_m=pos,shaft_error_rad=angle,failure=(pos>.05)|(angle>=np.pi/3),
                position_reward=torch.exp(-(pos/.05).square()))
