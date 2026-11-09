# SPDX-License-Identifier: Apache-2.0
# Camera model adapted from Wuji Technology Dex-Supermimic 2becb738,
# tasks/pen_spin/mdp/observations.py. Compact equivalent state replaces its ring buffer.
"""Simulated camera, not a trained RGB/depth pose estimator.

Stores previous and held noisy measurements (not privileged fresh poses).
The same functional transition is usable for real and imagined sensor histories.
"""
import torch
from unionmimic.motions import qmul

def axis_rotation(angle,axis):
    axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    return torch.cat(((angle/2).cos(),(angle/2).sin()*axis),-1)

class PenCamera:
    def __init__(self,n,device):
        self.bias_pos=torch.zeros(n,3,device=device);self.bias_quat=torch.zeros(n,4,device=device);self.bias_quat[:,0]=1
        self.previous=self.bias_quat.new_zeros(n,7);self.previous[:,3]=1;self.held=self.previous.clone()
        self.hold_ticks=torch.zeros(n,dtype=torch.long,device=device);self.ready=torch.zeros(n,dtype=torch.bool,device=device)
    def reset(self,ids,pose=None):
        dev=self.previous.device;n=len(ids)
        self.bias_pos[ids]=(torch.rand(n,3,device=dev)*2-1)*.008
        self.bias_quat[ids]=axis_rotation((torch.rand(n,1,device=dev)*2-1)*.05,torch.randn(n,3,device=dev))
        self.ready[ids]=False;self.hold_ticks[ids]=0
    @staticmethod
    def transition(pose,bias_pos,bias_quat,previous,held,hold_ticks,ready,*,jitter_pos,jitter_quat,roll):
        measured=torch.cat((pose[:,:3]+bias_pos+jitter_pos,qmul(jitter_quat,qmul(bias_quat,pose[:,3:]))),1)
        q=measured[:,3:];measured=torch.cat((measured[:,:3],torch.where(q[:,:1]<0,-q,q)),1)
        holding=torch.where(hold_ticks>0,(roll<.78)&(hold_ticks<40),roll<.03)
        ticks=torch.where(holding,hold_ticks+1,torch.zeros_like(hold_ticks))
        observed=torch.where(ready[:,None],torch.where(holding[:,None],held,previous),measured)
        return observed,measured,observed.clone(),ticks,torch.ones_like(ready)
    def advance(self,current):
        n=len(current);dev=current.device
        args=dict(jitter_pos=(torch.rand(n,3,device=dev)*2-1)*.002,
            jitter_quat=axis_rotation((torch.rand(n,1,device=dev)*2-1)*.015,torch.randn(n,3,device=dev)),roll=torch.rand(n,device=dev))
        obs,self.previous,self.held,self.hold_ticks,self.ready=self.transition(current,self.bias_pos,self.bias_quat,
            self.previous,self.held,self.hold_ticks,self.ready,**args)
        return obs
    def snapshot(self):
        return {k:getattr(self,k).clone() for k in ('bias_pos','bias_quat','previous','held','hold_ticks','ready')}
    def restore(self,state):
        for k,v in state.items():setattr(self,k,v.clone())
