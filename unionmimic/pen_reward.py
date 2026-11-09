"""Shared Wuji tracking/actuator reward terms for Sharpa, in reward per second.

Source: Dex-Supermimic 2becb738, pen_spin/mdp/rewards.py. Self-collision
count is deliberately not inferred from net contact force: the latter includes
valid pen contacts. That missing pair sensor remains a separate acceptance item.
"""
import torch
from unionmimic.pen_tasks import LINK_NAMES
from unionmimic.motions import tracking_metrics


def update_charge(charge,torque,limits,dt=.02):
    load=(torque.abs()/limits).clamp(max=1.)
    return (charge+dt*(load-charge/2)).clamp(0,1)


def reward_terms(tasks,pose,q,torque,charge,action,previous,older,task_ids,clip_ids,steps,lower,upper,effort):
    goal=tasks.target(task_ids,clip_ids,steps);metrics=tracking_metrics(pose,goal['pose'])
    points=(tasks.endpoints(pose)-tasks.endpoints(goal['pose'])).reshape(-1,2,3)
    axis_error=points.norm(dim=-1).mean(-1)
    active=(task_ids>=2).float()
    links=tasks.fk.forward(q,LINK_NAMES)
    terms=dict(object_pos=2*torch.exp(-(metrics['position_error_m']/.05).square()),
        object_rot=8*torch.exp(-(axis_error/.04).square()),
        joint_pos=active*torch.exp(-((q-goal['q'])/(upper-lower).clamp_min(1e-6)/.1).square()).mean(-1),
        link_pos=active*torch.exp(-((links-goal['links']).norm(dim=-1)/.025).square()).mean(-1),
        action_rate=-.1*((action-previous).square()+(action-2*previous+older).square()).sum(-1),
        torque=-(torque/effort).square().sum(-1),
        overcurrent=-4*(charge-.8).clamp_min(0).sum(-1)/.2)
    return terms,metrics['failure'],goal['ended'],metrics
