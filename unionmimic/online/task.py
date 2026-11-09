"""Tensor contract for the existing Sharpa task (XY Z W quaternions).

Reward mirrors wm_craftnet_rotation.compute_reward/compute_hand_reward_finger;
runtime parity is mandatory. Hard contact/termination decisions stay hard.
The failure classifier supplies differentiable survival during imagination.
"""
import torch
from torch import nn
from .core import mlp


def layout(fields):
    result, offset = {}, 0
    for name, size in fields:
        result[name] = slice(offset, offset + size)
        offset += size
    return result, offset


S, DS = layout([('q',22), ('v',22), ('object',13), ('tips',15), ('force',5),
                ('palm',1), ('thumb',1), ('torque',22), ('fd',22), ('external_force',3),
                ('target',22), ('action',22), ('last_contact',5), ('stall',1), ('progress',1)])
C, DC = layout([('initial_pos',3), ('initial_quat',4), ('goal',3), ('axis',3),
                ('mass',1), ('friction',3), ('pgain',22), ('dgain',22), ('gravity',3),
                ('threshold',5), ('penalty_scale',1), ('action_corr',22), ('action_std',1)])
DYNAMIC = S['target'].start


def qmul(a, b):
    return torch.cat((a[...,3:] * b[...,:3] + b[...,3:] * a[...,:3]
                      + torch.cross(a[...,:3], b[...,:3], dim=-1),
                      a[...,3:] * b[...,3:] - (a[...,:3]*b[...,:3]).sum(-1, keepdim=True)), -1)


def qconj(q):
    return torch.cat((-q[...,:3], q[...,3:]), -1)


def qrotate(q, v):
    t = 2 * torch.cross(q[...,:3], v, dim=-1)
    return v + q[...,3:] * t + torch.cross(q[...,:3], t, dim=-1)


def spin_delta(a, b, axis):
    d = nn.functional.normalize(qmul(b, qconj(a)), dim=-1)
    d = torch.where(d[...,3:] < 0, -d, d)
    n = d[...,:3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(n, d[...,3:].clamp_min(1e-8))
    rot = d[...,:3] * torch.where(n > 1e-7, angle / n.clamp_min(1e-7), 2*torch.ones_like(n))
    along = (rot * axis).sum(-1)
    off = (rot - along[:,None] * axis).norm(dim=-1)
    return along, off


class HandTask(nn.Module):
    def __init__(self, constants):
        super().__init__()
        self.cfg = constants
        for name in ['lower', 'upper', 'default_q', 'pose_weights']:
            self.register_buffer(name, torch.tensor(constants[name], dtype=torch.float))

    def controller(self, s, command, scale):
        action = .8 * command.clamp(-1, 1) + .2 * s[:,S['action']]
        target = (s[:,S['target']] + scale * action).maximum(self.lower).minimum(self.upper)
        return action, target

    def imagined_command(self, command, ctx):
        return (command.clamp(-1,1) + ctx[:,C['action_corr']]
                + torch.randn_like(command)*ctx[:,C['action_std']]).clamp(-1,1)

    def reward(self, s, sn, ctx):
        cfg = self.cfg
        ob, old = sn[:,S['object']], s[:,S['object']]
        pos, quat, vel, angular = ob[:,:3], ob[:,3:7], ob[:,7:10], ob[:,10:13]
        axis = ctx[:,C['axis']]
        along, off = spin_delta(old[:,3:7], quat, axis)
        distance = (pos[:,None] - sn[:,S['tips']].reshape(-1,5,3)).norm(dim=-1)
        dpen = ((distance-cfg['distance_deadzone']).clamp_min(0)/cfg['distance_width']).square().clamp(0,1).mean(-1)
        force = sn[:,S['force']].clamp_min(0)
        contacts = (force >= cfg['tac_thresh']).float().sum(-1).clamp(0,5)
        err = (sn[:,S['target']] - sn[:,S['q']]).norm(dim=-1)
        pose = (((sn[:,S['q']]-self.default_q).square()) * self.pose_weights).sum(-1)
        scale = ctx[:,C['penalty_scale']].squeeze(-1)
        stall = torch.where((angular*axis).sum(-1).abs() < cfg['stall_speed'],
                            s[:,S['stall']].squeeze(-1)+1, torch.zeros_like(along))
        fall = (pos-ctx[:,C['goal']]).norm(dim=-1) >= cfg['fall_dist']
        initial_axis = qrotate(qconj(ctx[:,C['initial_quat']]), axis)
        current_axis = qrotate(quat, initial_axis)
        tilt = torch.acos((current_axis*axis).sum(-1).clamp(-1,1))
        failed = fall | (tilt > .4 * 3.1415926) | (stall >= cfg['stall_steps'])
        timeout = sn[:,S['progress']].squeeze(-1) >= cfg['episode_steps']-1
        reward = (cfg['spin_coef']*(along*20).clamp(-6.28,cfg['max_spin'])
                  + cfg['vel_coef']*vel.norm(dim=-1) + cfg['contact_coef']*contacts
                  - cfg['finger_coef']*dpen - cfg['axis_coef']*(off*20).square()
                  + scale*cfg['torque_coef']*sn[:,S['torque']].square().sum(-1)
                  + scale*cfg['work_coef']*(sn[:,S['torque']].abs()*sn[:,S['fd']].abs()).sum(-1)
                  + scale*cfg['action_coef']*sn[:,S['action']].square().sum(-1)
                  + scale*cfg['control_coef']*err + cfg['pose_coef']*torch.exp(-pose)
                  - cfg['position_coef']*(pos-ctx[:,C['initial_pos']]).square().sum(-1)
                  + (stall >= cfg['stall_steps']).float()*cfg['stall_penalty'])
        reward = torch.nan_to_num(reward, nan=-1e3, posinf=1e3, neginf=-1e3)
        reward = reward + fall.float()*cfg['fall_penalty']
        # Exact extra native task reward terms; not changed for model convenience.
        excess = (force-cfg['tip_threshold']).relu().mean(-1)
        reward = reward - cfg['tip_coef']*(torch.exp(cfg['tip_alpha']*excess)-1)
        reward = reward + cfg['palm_coef']*(sn[:,S['palm']].squeeze(-1)>.5).float()
        reward = reward - cfg['thumb_penalty']*(sn[:,S['thumb']].squeeze(-1)>cfg['thumb_threshold']).float()
        return reward, failed, timeout, stall, dict(rotation=along, offaxis=off, tilt=tilt)

    def observation(self, sn, history, ctx, stochastic=True):
        q = 2*(sn[:,S['q']]-self.lower)/(self.upper-self.lower)-1
        if stochastic:
            q = q + (torch.rand_like(q)-.5)*.12
        target = 2*(sn[:,S['target']]-self.lower)/(self.upper-self.lower)-1
        contacts = (sn[:,S['force']] >= ctx[:,C['threshold']]).float()
        if stochastic:
            lag = torch.rand_like(contacts) < self.cfg['latency']
            contacts = torch.where(lag, sn[:,S['last_contact']], contacts)
            sensed = contacts * (torch.rand_like(contacts) >= self.cfg['sensor_noise']).float()
        else:
            sensed = contacts
        frame = torch.cat((q, target, sensed, ctx[:,C['axis']].repeat(1,8)), -1)
        return torch.cat((frame, history[:,:-73]),-1).clamp(-5,5), contacts


class PhysicalModel(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.net = mlp(DS+DC+44, (512,512,512), DYNAMIC, True)
        self.term = mlp(DS+DC, (128,128), 1, True)
        self.register_buffer('s_mean',torch.zeros(DS)); self.register_buffer('s_std',torch.ones(DS))
        self.register_buffer('c_mean',torch.zeros(DC)); self.register_buffer('c_std',torch.ones(DC))
        self.register_buffer('d_mean',torch.zeros(DYNAMIC)); self.register_buffer('d_std',torch.ones(DYNAMIC))
        # Start near the persistence mean, not arbitrary physical magnitudes.
        nn.init.uniform_(self.net[-1].weight,-1e-3,1e-3); nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def fit_stats(self, s, sn, ctx):
        for x,name,floor in [(s,'s',.01),(ctx,'c',.01),(sn[:,:DYNAMIC]-s[:,:DYNAMIC],'d',.001)]:
            getattr(self,name+'_mean').copy_(x.mean(0))
            getattr(self,name+'_std').copy_(x.std(0,unbiased=False).clamp_min(floor))

    def forward(self, s, command, ctx, scale):
        action,target = self.task.controller(s, command, scale)
        norm_s=((s-self.s_mean)/self.s_std).clamp(-20,20)
        norm_c=((ctx-self.c_mean)/self.c_std).clamp(-20,20)
        target_n=2*(target-self.task.lower)/(self.task.upper-self.task.lower)-1
        delta=self.net(torch.cat((norm_s,norm_c,action,target_n),-1))*self.d_std+self.d_mean
        physical=s[:,:DYNAMIC]+delta
        # Quaternion normalization preserves gradients and the rotation manifold.
        qi=S['object'].start+3
        physical=torch.cat((physical[:,:qi],nn.functional.normalize(physical[:,qi:qi+4],dim=-1),physical[:,qi+4:]),-1)
        sn=torch.cat((physical,target,action,s[:,S['last_contact']],s[:,S['stall']],s[:,S['progress']]+1),-1)
        logit=self.term(torch.cat((((sn-self.s_mean)/self.s_std).clamp(-20,20),norm_c),-1)).squeeze(-1)
        return sn, logit

    def error(self, pred, truth):
        qi=S['object'].start+3
        truth_q=truth[:,qi:qi+4]
        truth_q=torch.where((pred[:,qi:qi+4]*truth_q).sum(-1,keepdim=True)<0,-truth_q,truth_q)
        aligned=torch.cat((truth[:,:qi],truth_q,truth[:,qi+4:DYNAMIC]),-1)
        return ((pred[:,:DYNAMIC]-aligned)/self.s_std[:DYNAMIC]).square().mean()
