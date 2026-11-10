"""Sharpa-specific sensors, controller memory, native rewards and observations."""
import torch
from torch import nn
from in_hand_rotation_mjlab.world_model.model import WorldModel, WMConfig, StateError, mlp
from unionmimic.sharpa_leap import L, physical_state, object_root_from_physical, SharpaCommandMap
from unionmimic.online_baseline.task import HandTask, S, C, qrotate, qconj

X=dict(tips=slice(79,94),force=slice(94,99),palm=slice(99,100),thumb=slice(100,101),
       torque=slice(101,123),fd=slice(123,145),external_force=slice(145,148),
       action=slice(148,170),last_contact=slice(170,175),stall=slice(175,176),progress=slice(176,177))
DX=177;DC=100;H=10


def pack(legacy,base):
    core=physical_state(legacy[:,S['q']],legacy[:,S['v']],legacy[:,S['target']],legacy[:,S['object']],base)
    tips=legacy[:,S['tips']].reshape(-1,5,3)
    local=qrotate(qconj(base[:,None,3:7]).expand(-1,5,-1),tips-base[:,None,:3]).flatten(1)
    return torch.cat([core,local]+[legacy[:,S[k]] for k in ('force','palm','thumb','torque','fd','external_force','action','last_contact','stall','progress')],-1)


def legacy_state(s,ctx):
    base=torch.cat((ctx[:,93:100],torch.zeros(len(s),6,device=s.device)),1)
    obj=object_root_from_physical(s[:,:79],base)
    tips=qrotate(base[:,None,3:7].expand(-1,5,-1),s[:,X['tips']].reshape(-1,5,3))+base[:,None,:3]
    return torch.cat([s[:,L['q']],s[:,L['qd']],obj,tips.flatten(1)]+[s[:,X[k]] for k in ('force','palm','thumb','torque','fd','external_force')]+
                     [s[:,L['cmd']]]+[s[:,X[k]] for k in ('action','last_contact','stall','progress')],-1)


class SharpaModel(nn.Module):
    def __init__(self,constants):
        super().__init__();self.S=L;self.constants=constants
        self.physics=WorldModel(WMConfig(nj=22,dc=DC+3,dt=constants['control_dt']))
        # The native yaw failure feature lacks Sharpa's reset-relative/stall context.
        # Reuse its physical integration unchanged, with a task-complete failure head.
        self.physics.term=nn.Identity()
        self.controller=SharpaCommandMap(constants['lower'],constants['upper'])
        self.sensors=mlp(self.physics.din,(512,512),66,True)
        self.term=mlp(2*22+15+DC+1,(256,256),1,True)
        for name,n in [('c_mu',DC),('c_sd',DC),('sensor_mu',self.physics.din),('sensor_sd',self.physics.din),
                       ('y_mu',66),('y_sd',66),('term_mu',2*22+15+DC+1),('term_sd',2*22+15+DC+1)]:
            self.register_buffer(name,torch.ones(n) if name.endswith('sd') else torch.zeros(n))

    def normalize(self,c):return (c-self.c_mu)/self.c_sd
    def raw_context(self,c):return c*self.c_sd+self.c_mu
    def physics_context(self,s,c,force=None):return torch.cat((c,s[:,X['external_force']] if force is None else force),-1)
    def sensor_features(self,s,sn,c,force=None):
        if force is None:force=sn[:,X['external_force']]
        return self.physics._feat(sn[:,:79],sn[:,L['cmd']],self.physics_context(s,c,force))
    def failure_features(self,s,c):
        return torch.cat((self.physics._term_feat(s[:,:79]),c,s[:,X['stall']]/self.constants['stall_steps']),-1)
    def failure_logit(self,s,c):return self.term((self.failure_features(s,c)-self.term_mu)/self.term_sd).squeeze(-1)
    def sensor_raw(self,s,sn,c,force=None):return self.sensors((self.sensor_features(s,sn,c,force)-self.sensor_mu)/self.sensor_sd)
    def sensor_loss(self,s,sn,c):
        pred=self.sensor_raw(s,sn,c);target=sn[:,79:145]
        # LEAP ContactHead uses positive forces and log-force SmoothL1, avoiding
        # enormous normalization weights when warm-up has no palm/thumb contact.
        force=nn.functional.softplus(pred[:,15:22])*2
        lf=nn.functional.smooth_l1_loss(torch.log1p(force),torch.log1p(target[:,15:22].clamp_min(0)))
        error=pred-(target-self.y_mu)/self.y_sd
        return lf+torch.cat((error[:,:15],error[:,22:]),-1).square().mean()

    def next_force(self,s,c):
        old=s[:,X['external_force']];scale=self.constants.get('force_scale',0.)
        if scale<=0:return old
        raw=self.raw_context(c);mass=raw[:,C['mass']]
        changed=torch.rand(len(s),1,device=s.device)<self.constants['force_probability']
        return torch.where(changed,torch.randn_like(old)*mass*scale,old*self.constants['force_decay'])

    @torch.no_grad()
    def fit_stats(self,s,sn,c):
        self.c_mu.copy_(c.mean(0));self.c_sd.copy_(c.std(0).clamp_min(1e-6));nc=self.normalize(c)
        self.physics.fit_stats(s[:,:79],sn[:,L['cmd']],self.physics_context(s,nc,sn[:,X['external_force']]),sn[:,:79])
        for data,prefix in [(self.sensor_features(s,sn,nc),'sensor'),(sn[:,79:145],'y'),(self.failure_features(sn,nc),'term')]:
            getattr(self,prefix+'_mu').copy_(data.mean(0));getattr(self,prefix+'_sd').copy_(data.std(0).clamp_min(1e-4))

    def _physics_step(self,s,command,c,scale,external_force):
        if scale is None:scale=self.constants['relative_scale']*(1+(torch.rand(len(s),1,device=s.device)-.5)*.1)
        target,action=self.controller(s[:,L['cmd']],command,s[:,X['action']],scale)
        force=self.next_force(s,c) if external_force is None else external_force
        core,_=self.physics(s[:,:79],target,self.physics_context(s,c,force))
        raw=self.raw_context(c)
        # Core angular velocity is in fixed hand coordinates; task axis is world-frame.
        omega=qrotate(raw[:,96:100],core[:,L['w']])
        stall=torch.where((omega*raw[:,C['axis']]).sum(-1,keepdim=True).abs()<self.constants['stall_speed'],
                          s[:,X['stall']]+1,torch.zeros_like(s[:,X['stall']]))
        return core,action,force,stall

    def physical_transition(self,s,command,c,*,scale,external_force):
        """Advance the fields read by physics and failure supervision during fitting."""
        core,action,force,stall=self._physics_step(s,command,c,scale,external_force)
        return torch.cat((core,s[:,79:145],force,action,s[:,X['last_contact']],stall,s[:,X['progress']]+1),-1)

    def forward(self,s,command,c,scale=None,external_force=None):
        core,action,force,stall=self._physics_step(s,command,c,scale,external_force)
        pred=self.sensor_raw(s,core,c,force);aux=pred*self.y_sd+self.y_mu
        aux=torch.cat((aux[:,:15],nn.functional.softplus(pred[:,15:22])*2,aux[:,22:],force),-1)
        nxt=torch.cat((core,aux,action,s[:,X['last_contact']],stall,s[:,X['progress']]+1),-1)
        return nxt,self.failure_logit(nxt,c)


class PhysicalError(StateError):
    def __init__(self):super().__init__(22)
    def fit(self,s,sn):return super().fit(s[:,:79],sn[:,:79])
    def forward(self,s,sn):return super().forward(s[:,:79],sn[:,:79])


class SharpaTask:
    def __init__(self,model,constants,device):
        self.model=model;self.native=HandTask(constants).to(device);self.cfg=constants
    def flatten(self,frames):return frames.flatten(1)
    def critic_features(self,s,info,dprev,c):
        return torch.cat((self.model.physics._term_feat(s[:,:79]),s[:,L['cmd']]-s[:,L['q']],s[:,79:],c),-1)
    def imagined_command(self,a,c):return self.native.imagined_command(a,self.model.raw_context(c)[:,:93])
    def reward_rates(self,s,sn,info,dprev,c):
        raw=self.model.raw_context(c);r,*_=self.native.reward(legacy_state(s,raw),legacy_state(sn,raw),raw[:,:93])
        return {'native':r/self.cfg['control_dt']},dprev
    def imagined_frame(self,sn,c):
        raw=self.model.raw_context(c)
        frame,contacts=self.native.observation(legacy_state(sn,raw),torch.zeros(len(sn),73,device=sn.device),raw[:,:93],stochastic=True)
        sn=torch.cat((sn[:,:170],contacts,sn[:,175:]),-1)
        return frame,sn
    def max_step_reward(self):
        c=self.cfg
        for k in ('vel_coef','torque_coef','work_coef','action_coef','control_coef','stall_penalty','fall_penalty'):assert c[k]<=0,k
        for k in ('finger_coef','axis_coef','position_coef','tip_coef','thumb_penalty'):assert c[k]>=0,k
        return max(c['spin_coef']*-6.28,c['spin_coef']*c['max_spin'])+max(0,c['contact_coef']*5)+max(0,c['pose_coef'])+max(0,c['palm_coef'])
