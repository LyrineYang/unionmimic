"""Shared four-action targets and observable pose contract (wxyz, hand frame).

Only known reference targets enter future goal features. Actual pose measurements
have per-episode bias, per-step noise and one control-step latency. This module
is independent of the policy algorithm and provides the imagination interface.
"""
import json, math,hashlib
from pathlib import Path
import torch
from unionmimic.motions import qmul,qrot,qinv,qslerp,tracking_metrics
from unionmimic.hand_kinematics import HandKinematics

TASKS=('reorient_hold','perturb_recover','reference_single_axis','reference_multi_axis')
ROOT=Path(__file__).resolve().parents[1]
LINK_NAMES=['right_hand_C_MC']+[
    'right_'+f+'_'+suffix for f in ('thumb','index','middle','ring','pinky')
    for suffix in (('CMC_VL','MC','PP','DP') if f=='thumb' else ('MCP_VL','PP','MP','DP'))]

def difference(x,dt=.02):
    out=torch.zeros_like(x)
    if len(x)>1:out[:-1]=(x[1:]-x[:-1])/dt;out[-1]=out[-2]
    return out

def pose_velocity(pose,dt=.02):
    """Wuji commands.object_velocity_from_pose: velocities in the fixed hand frame."""
    delta=qmul(pose[1:,3:],qinv(pose[:-1,3:]));delta=torch.nn.functional.normalize(delta,dim=-1)
    delta=torch.where(delta[:,:1]<0,-delta,delta);s=delta[:,1:].norm(dim=-1,keepdim=True)
    angular=delta[:,1:]/s.clamp_min(1e-8)*(2*torch.atan2(s,delta[:,:1].clamp(-1,1))/dt)
    angular=torch.cat((angular,angular[-1:]),0)
    return difference(pose[:,:3],dt),angular

class PenTasks:
    DT=.02
    def __init__(self,root,device):
        root=Path(root);self.manifest=json.loads((root/'manifest.json').read_text());self.clips=[]
        for info in self.manifest['clips']:
            if 'retarget_sha256' in info:
                assert hashlib.sha256((root/info['file']).read_bytes()).hexdigest()==info['retarget_sha256'],'Retarget clip changed after selection'
            c=torch.load(root/info['file'],map_location=device);c.update(id=info['id'],frames=info['frames'])
            assert c['frame_dt']==self.DT and torch.isfinite(c['q']).all() and torch.isfinite(c['pose']).all()
            self.clips.append(c)
        self.lengths=torch.tensor([len(c['q']) for c in self.clips],device=device)
        self.offsets=torch.cat((torch.zeros(1,dtype=torch.long,device=device),self.lengths.cumsum(0)[:-1]))
        self.fk=HandKinematics(ROOT/'work/upstreams/WM-Craftnet/assets/urdf/right_sharpa_wave/right_sharpa_wave.urdf',self.manifest['joint_names'],device)
        for c in self.clips:
            c['qd']=difference(c['q']);c['v'],c['w']=pose_velocity(c['pose'])
            c['links']=self.fk.forward(c['q'],LINK_NAMES)
        self.flat={k:torch.cat([c[k] for c in self.clips]) for k in ('q','qd','pose','v','w','links')}
        self.initial=torch.stack([c['pose'][0] for c in self.clips]);self.initial_q=torch.stack([c['q'][0] for c in self.clips])
        self.device=device
        self.single=torch.tensor([i for i,c in enumerate(self.clips) if c['id'].startswith('02_')],device=device)
        self.multi=torch.tensor([i for i,c in enumerate(self.clips) if c['id'].startswith('03_')],device=device)

    def choose(self,task_ids):
        clips=torch.empty_like(task_ids)
        for task in range(4):
            mask=task_ids==task;pool=self.multi if task==3 else self.single
            clips[mask]=pool[torch.randint(len(pool),(int(mask.sum()),),device=self.device)]
        return clips

    def target(self,task_ids,clip_ids,steps):
        seconds=steps.float()*self.DT;isref=task_ids>=2
        t=torch.where(isref,steps.float(),torch.zeros_like(seconds)).clamp_min(0)
        t=torch.minimum(t,(self.lengths[clip_ids]-1).float());lo=t.floor().long();hi=torch.minimum(lo+1,self.lengths[clip_ids]-1)
        u=(t-lo)[:,None];lo=lo+self.offsets[clip_ids];hi=hi+self.offsets[clip_ids]
        out={k:(1-u)*self.flat[k][lo]+u*self.flat[k][hi] for k in ('q','qd','v','w')}
        a,b=self.flat['pose'][lo],self.flat['pose'][hi]
        pose=torch.cat(((1-u)*a[:,:3]+u*b[:,:3],qslerp(a[:,3:],b[:,3:],u)),-1)
        out['links']=(1-u[:,:,None])*self.flat['links'][lo]+u[:,:,None]*self.flat['links'][hi]
        moving=isref&(steps>=0)&(steps<self.lengths[clip_ids]-1)
        for k in ('qd','v','w'):out[k]=out[k]*moving[:,None]
        ended=torch.where(isref,steps>=self.lengths[clip_ids]-1,seconds>=6.)
        # C1: a fixed 60-degree wrist-Y reorientation over 2 seconds, then hold.
        mask=task_ids==0;theta=(seconds[mask]/2).clamp(0,1)*math.pi/3
        delta=torch.stack(((theta/2).cos(),torch.zeros_like(theta),(theta/2).sin(),torch.zeros_like(theta)),-1)
        pose[mask,3:]=qmul(delta,pose[mask,3:])
        out.update(pose=pose,ended=ended);return out

    @staticmethod
    def endpoints(pose,bounded=False):
        axis=torch.zeros_like(pose[...,:3]);axis[...,0]=.10
        offset=qrot(pose[...,3:],axis)
        points=torch.cat((pose[...,:3]+offset,pose[...,:3]-offset),-1)
        return .15*torch.tanh(points/.15) if bounded else points

    def goal_features(self,task_ids,clip_ids,steps):
        target=self.target(task_ids,clip_ids,steps)
        # Port the reference command: qpos + bounded pen points + retargeted
        # link positions. C1/C2 have object goals, without a scored joint motion.
        active=(task_ids>=2)[:,None]
        return torch.cat((target['q']*active,self.endpoints(target['pose'],bounded=True),
                          target['links'].flatten(1)*active,torch.nn.functional.one_hot(task_ids,4).float()),-1)

    @staticmethod
    def perturbation(task_ids,steps):
        # Known C2 body-local +Y pulse: 0.15 N for 0.20 s at t=2 s.
        force=torch.zeros(len(steps),3,device=steps.device)
        force[:,1]=((task_ids==1)&(steps>=100)&(steps<110)).float()*.15
        return force

    def reward(self,pose,task_ids,clip_ids,steps,action):
        goal=self.target(task_ids,clip_ids,steps);m=tracking_metrics(pose,goal['pose'])
        # Wuji object tracking terms, weights/sigmas unchanged. Physics probe
        # uses this subset only; learner promotion additionally requires the
        # audited hardware penalties and retargeted joint/link reward contract.
        points=(self.endpoints(pose)-self.endpoints(goal['pose'])).reshape(-1,2,3)
        axis_error=points.norm(dim=-1).mean(-1)
        rate=2*torch.exp(-(m['position_error_m']/.05).square())+8*torch.exp(-(axis_error/.04).square())
        reward=rate*self.DT
        return reward,m['failure'],goal['ended'],m

class ResidualCommand(torch.nn.Module):
    """Wuji SingleHandResidualEMAAction: scale .5, residual clip .6, alpha .5.

    The target is limited to Sharpa's physical control range. Reference motions
    are known task goals available equally to both algorithms, not policy weights.
    """
    def __init__(self,lower,upper):
        super().__init__();self.register_buffer('lower',torch.as_tensor(lower));self.register_buffer('upper',torch.as_tensor(upper))
    def forward(self,reference_q,command,previous_residual):
        residual=.5*(command*.5).clamp(-.6,.6)+.5*previous_residual
        target=(reference_q+residual).maximum(self.lower).minimum(self.upper)
        return target,residual
