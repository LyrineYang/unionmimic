"""Read-only local action-response probes, with a measured restore-noise floor.

Isaac Gym does not expose the contact solver's entire hidden state. Never call a
tensor restore exact: compare repeated zero-perturbation branches first.
"""
import random
import numpy as np
import torch
from isaacgym import gymtorch
from .collect import terminal_flags


def snapshot_object(obj):
    result = {}
    for key, value in vars(obj).items():
        if torch.is_tensor(value):
            result[key] = ('tensor', value.detach().clone())
        elif isinstance(value, (int, float, bool)):
            result[key] = ('scalar', value)
        elif isinstance(value, dict):
            tensors = {k:v.detach().clone() for k,v in value.items() if torch.is_tensor(v)}
            if tensors: result[key] = ('dict', tensors)
    return result


def restore_object(obj, snapshot):
    for key, (kind, value) in snapshot.items():
        if kind == 'tensor':
            current=getattr(obj,key,None)
            if torch.is_tensor(current) and current.shape == value.shape:
                current.copy_(value)
            else: setattr(obj,key,value.clone())
        elif kind == 'scalar': setattr(obj,key,value)
        else:
            current=getattr(obj,key)
            for k,v in value.items(): current[k]=v.clone()


def snapshot(env):
    # The noise closures capture these nested dictionaries; restore in place.
    noise={k:{n:v.clone() for n,v in d.items() if torch.is_tensor(v)}
           for k,d in env.dr_randomizations.items() if isinstance(d,dict)}
    return dict(env=snapshot_object(env),adapter=snapshot_object(env.wm_adapter),noise=noise,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(env.device))


@torch.no_grad()
def restore(env, saved):
    restore_object(env,saved['env']);restore_object(env.wm_adapter,saved['adapter'])
    for k,d in saved['noise'].items():
        for n,v in d.items():env.dr_randomizations[k][n]=v.clone()
    env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor))
    env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state))
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],env.device)


@torch.no_grad()
def physical_branch(env, player, commands):
    rows=[];live=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
    for command in commands:
        _,_,_,_=player.env_step(env,command.clone())
        flags=terminal_flags(env)
        rows.append(dict(reward=env.rew_buf.clone(),joint=env.last_obs_buf[:,:22].clone(),
                         spin=env.last_spin_delta_axis.clone()/20,
                         failure=flags['failure'].clone(),mask=live.clone(),
                         applied=env.applied_command.clone(),executed=env.executed_command.clone()))
        live &= ~env.reset_buf.bool()
    return {k:torch.stack([r[k] for r in rows]) for k in rows[0]}


def imagined_branch(wm,heads,initial,last_action,applied,perturbation,repeats=16,seed=101):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    state={k:v.detach().repeat_interleave(repeats,0) for k,v in initial.items()}
    last=last_action.detach().repeat_interleave(repeats,0)
    delta=perturbation.repeat_interleave(repeats,0)
    rewards=[];joints=[];failures=[];executed=[]
    for h in range(len(applied)):
        noisy=applied[h].detach().repeat_interleave(repeats,0)
        if h==0:noisy=(noisy+delta).clamp(-1,1)
        action=.8*noisy+.2*last
        state=wm.dynamics.img_step(state,action);feat=wm.dynamics.get_feat(state)
        rd,pd,_=heads.predictions(feat)
        rewards.append(rd.mode().reshape(len(last_action),repeats).mean(1))
        joints.append(pd.mode()[:,:22].reshape(len(last_action),repeats,22).mean(1))
        failures.append(heads.failure_probability(feat).reshape(len(last_action),repeats).mean(1))
        executed.append(action.reshape(len(last_action),repeats,22).mean(1));last=action
    return dict(reward=torch.stack(rewards),joint=torch.stack(joints),failure=torch.stack(failures),
                executed=torch.stack(executed))
