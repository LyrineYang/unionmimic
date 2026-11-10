"""Pen task adapter around the pinned LEAP physical model and update functions.

Only task state, controller, sensors, reward and failure features change. The
physical network, integration, actor/critic architectures and optimizers do not.
No future measured frame is accepted by the imagined rollout interface.
"""
import torch
from torch import nn
from in_hand_rotation_mjlab.world_model.model import mlp
from unionmimic.sharpa_leap import L
from unionmimic.leap_sharpa.model import SharpaModel,legacy_state,X
from unionmimic.online_baseline.task import HandTask
from unionmimic.pen_tasks import ResidualCommand
from unionmimic.pen_camera import PenCamera,axis_rotation
from unionmimic.pen_reward import reward_terms,update_charge

DX=259;DC=109;DF=174;H=10
P=dict(previous=slice(177,184),held=slice(184,191),hold_ticks=slice(191,192),ready=slice(192,193),
       raw_previous=slice(193,215),raw_older=slice(215,237),charge=slice(237,259))


def constants(env):
    out={k:getattr(env,v).tolist() for k,v in dict(lower='arm_hand_dof_lower_limits',upper='arm_hand_dof_upper_limits',
         default_q='arm_hand_dof_default_pos',pose_weights='hand_pose_joint_weight',effort='arm_hand_dof_effort_limits').items()}
    out.update(control_dt=.02,relative_scale=.5,stall_steps=1,latency=env.latency,sensor_noise=env.sensor_noise,tac_thresh=env.tac_thresh)
    return out


class PenModel(SharpaModel):
    def physical_transition(self,s,command,c,*,scale,external_force):
        return self(s,command,c,scale=scale,external_force=external_force)[0]

    def __init__(self,cfg,tasks):
        super().__init__(cfg);self.tasks=tasks
        self.controller=ResidualCommand(cfg['lower'],cfg['upper'])
        self.c_mu=torch.zeros(DC);self.c_sd=torch.ones(DC)
        self.term=mlp(255,(256,256),1,True);self.term_mu=torch.zeros(255);self.term_sd=torch.ones(255)
        self.register_buffer('effort',torch.tensor(cfg['effort']))
    def physics_context(self,s,c,force=None):
        return torch.cat((c[:,:100],s[:,X['external_force']] if force is None else force),-1)
    def labels(self,c):
        raw=self.raw_context(c)
        return raw[:,100].round().long(),raw[:,101].round().long()
    def phase(self,s):return s[:,176]
    def goal(self,s,c):return self.tasks.goal_features(*self.labels(c),self.phase(s))
    def failure_features(self,s,c):
        return torch.cat((self.physics._term_feat(s[:,:79]),c[:,:100],self.goal(s,c),self.phase(s)[:,None]*.02),-1)
    def next_force(self,s,c):return self.tasks.perturbation(self.labels(c)[0],self.phase(s))
    def forward(self,s,command,c,scale=None,external_force=None):
        reference=self.tasks.target(*self.labels(c),self.phase(s))
        target,residual=self.controller(reference['q'],command,s[:,X['action']])
        force=self.next_force(s,c) if external_force is None else external_force
        core,_=self.physics(s[:,:79],target,self.physics_context(s,c,force))
        pred=self.sensor_raw(s,core,c,force);aux=pred*self.y_sd+self.y_mu
        aux=torch.cat((aux[:,:15],nn.functional.softplus(pred[:,15:22])*2,aux[:,22:],force),-1)
        base=torch.cat((core,aux,residual,s[:,X['last_contact']],torch.zeros_like(s[:,X['stall']]),s[:,X['progress']]+1),-1)
        n=len(s);raw=self.raw_context(c)
        _,previous,held,ticks,ready=PenCamera.transition(core[:,66:73],raw[:,102:105],raw[:,105:109],
            s[:,P['previous']],s[:,P['held']],s[:,191].long(),s[:,192].bool(),
            jitter_pos=(torch.rand(n,3,device=s.device)*2-1)*.002,
            jitter_quat=axis_rotation((torch.rand(n,1,device=s.device)*2-1)*.015,torch.randn(n,3,device=s.device)),
            roll=torch.rand(n,device=s.device))
        charge=update_charge(s[:,P['charge']],base[:,X['torque']],self.effort)
        sn=torch.cat((base,previous,held,ticks[:,None].float(),ready[:,None].float(),command,s[:,P['raw_previous']],charge),-1)
        return sn,self.failure_logit(sn,c)


class PenTask:
    def __init__(self,model,cfg,device):
        self.model=model;self.cfg=cfg;self.native=HandTask(cfg).to(device);self.tasks=model.tasks
    def flatten(self,frames):return frames.flatten(1)
    def critic_features(self,s,info,dprev,c):
        return torch.cat((self.model.physics._term_feat(s[:,:79]),s[:,L['cmd']]-s[:,L['q']],s[:,79:],c,self.model.goal(s,c)),-1)
    def imagined_command(self,a,c):return self.native.imagined_command(a,self.model.raw_context(c)[:,:93])
    def reward(self,s,sn,c):
        terms,f,end,m=reward_terms(self.tasks,sn[:,66:73],sn[:,L['q']],sn[:,X['torque']],sn[:,P['charge']],
            sn[:,P['raw_previous']],s[:,P['raw_previous']],s[:,P['raw_older']],*self.model.labels(c),self.model.phase(sn),
            self.native.lower,self.native.upper,self.model.effort)
        return sum(terms.values())*.02,f,end,m
    def reward_rates(self,s,sn,info,dprev,c):
        reward,*_=self.reward(s,sn,c);return {'tracking_and_actuator':reward/.02},dprev
    def imagined_frame(self,sn,c):
        raw=self.model.raw_context(c)
        frame,contacts=self.native.observation(legacy_state(sn[:,:177],raw[:,:100]),torch.zeros(len(sn),73,device=sn.device),raw[:,:93],stochastic=True)
        sn=torch.cat((sn[:,:170],contacts,sn[:,175:]),-1)
        frame=torch.cat((frame,self.tasks.endpoints(sn[:,P['held']],bounded=True),self.model.goal(sn,c)),-1)
        assert frame.shape[-1]==DF
        return frame,sn
    def max_step_reward(self):return 12*.02
