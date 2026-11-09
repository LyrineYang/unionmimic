"""PPO control with the SM policy/critic interfaces and pinned LEAP PPO recipe.

Losses call WM-Craftnet's vendored rl_games primitives, equivalent to the
RSL-RL clipped objective. No model predictions enter PPO's rollout or returns.
"""
import math
import torch
from rl_games.common.common_losses import actor_loss,critic_loss
from in_hand_rotation_mjlab.world_model.model import quat_to_rot6d
from unionmimic.sharpa_leap import L
from unionmimic.online_baseline.core import gae

RECIPE=dict(envs=4096,rollout=32,epochs=5,minibatches=4,lr=1e-3,clip=.2,entropy=.003,
            gamma=.99,lam=.95,desired_kl=.01,max_grad=1.,initial_std=.7,min_std=.05,max_std=1.)

def features(s,c):
    physical=torch.cat([s[...,L[k]] if k!='quat' else quat_to_rot6d(s[...,L[k]])
                        for k in ('q','qd','p','quat','v','w')],-1)
    return torch.cat((physical,s[...,L['cmd']]-s[...,L['q']],s[...,79:],c),-1)

class PairedPPO:
    def __init__(self,actor,critic):
        self.actor,self.critic=actor,critic
        self.log_std=torch.nn.Parameter(torch.full((22,),math.log(.7),device=next(actor.parameters()).device))
        self.optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters())+[self.log_std],lr=RECIPE['lr'])
        self.steps=0

    def distribution(self,history):
        return torch.distributions.Normal(self.actor(history),self.log_std.exp())

    def update(self,rows):
        b={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
        adv,ret=gae(b['reward'],b['value'],b['next_value'],b['failure'],b['timeout'],b['valid'],.99,.95)
        mask=b['valid'].flatten();flat={k:v.flatten(0,1)[mask] for k,v in b.items()}
        adv=adv.flatten()[mask];ret=ret.flatten()[mask];adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        metrics=[]
        for _ in range(5):
            for ids in torch.randperm(len(adv),device=adv.device).chunk(4):
                dist=self.distribution(flat['history'][ids]);old=torch.distributions.Normal(flat['mean'][ids],flat['std'][ids])
                with torch.no_grad():
                    kl=float(torch.distributions.kl_divergence(old,dist).sum(-1).mean())
                    lr=self.optimizer.param_groups[0]['lr']
                    if kl>.02:lr=max(1e-5,lr/1.5)
                    elif 0<kl<.005:lr=min(1e-2,lr*1.5)
                    for g in self.optimizer.param_groups:g['lr']=lr
                # Likelihood is for the raw Gaussian draw, not the clipped command.
                logp=dist.log_prob(flat['raw_action'][ids]).sum(-1)
                pi=actor_loss(-flat['logp'][ids],-logp,adv[ids],True,.2).mean()
                v=critic_loss(flat['value'][ids],self.critic(flat['features'][ids]),.2,ret[ids],True).mean()
                loss=pi+v-.003*dist.entropy().sum(-1).mean()
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                p=list(self.actor.parameters())+[self.log_std];vg=list(self.critic.parameters())
                pn=torch.nn.utils.clip_grad_norm_(p,1.);vn=torch.nn.utils.clip_grad_norm_(vg,1.)
                assert torch.isfinite(loss) and torch.isfinite(pn) and torch.isfinite(vn),'Nonfinite PPO update'
                assert all(torch.isfinite(x.grad).all() for x in p+vg if x.grad is not None)
                self.optimizer.step()
                with torch.no_grad():self.log_std.clamp_(math.log(.05),0.)
                self.steps+=1;metrics.append((float(pi),float(v),kl,float(pn),float(vn)))
        # Freeze moments throughout each on-policy rollout and its updates.
        self.actor.norm.update(flat['history']);self.critic.norm.update(flat['features'])
        mean=torch.tensor(metrics).mean(0).tolist()
        return dict(policy_loss=mean[0],value_loss=mean[1],kl=mean[2],actor_grad=mean[3],critic_grad=mean[4],
                    lr=lr,std=float(self.log_std.exp().mean()),optimizer_steps=self.steps,valid_rows=int(mask.sum()))
