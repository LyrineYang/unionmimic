"""Independent F + BPTT + TD(lambda), and the native clipped PPO loss."""
import time
import torch
from torch import nn
from rl_games.common.common_losses import actor_loss, critic_loss
from rl_games.common.schedulers import AdaptiveScheduler
from .core import frozen, checked_step, lambda_returns, objective, gae, BudgetExpired, RunStopped
from unionmimic.online_baseline.task import S


def advance(task, model, s, history, ctx, command, scale, observations=True):
    sn,logit=model(s,command,ctx,scale)
    reward,fail,timeout,stall,metrics=task.reward(s,sn,ctx)
    if observations:
        history,contacts=task.observation(sn,history,ctx)
    else:
        contacts=sn[:,S['last_contact']]
    sn=torch.cat((sn[:,:S['last_contact'].start],contacts,stall[:,None],sn[:,S['progress']]),-1)
    return sn,history,reward*.01,logit,timeout,metrics


class Learner:
    def __init__(self, actor, critic, model, task, cfg):
        self.actor,self.critic,self.model,self.task,self.cfg=actor,critic,model,task,cfg
        self.log_std=nn.Parameter(torch.zeros(22,device=next(actor.parameters()).device))
        self.actor_opt=torch.optim.Adam(actor.parameters(),lr=cfg['actor_lr'],betas=(.9,.95))
        self.model_opt=torch.optim.Adam(model.parameters(),lr=cfg['model_lr'])
        self.critic_opt=torch.optim.Adam(critic.parameters(),lr=cfg['critic_lr'])
        self.ppo_opt=torch.optim.Adam(list(actor.parameters())+[self.log_std],lr=cfg['ppo_lr'])
        self.ppo_scheduler=AdaptiveScheduler(kl_threshold=.02)
        self.counts=dict(model=0,actor=0,critic=0,cycles=0)
        if not 0 <= cfg.get('failure_fraction', 0.) < 1:
            raise ValueError('failure_fraction must be in [0, 1)')
        if cfg.get('critic_norm_source', 'mixed') not in ('mixed', 'real'):
            raise ValueError('Unknown critic normalization source')
        self.deadline=float('inf')
        self.stop_requested=lambda:False

    def check_budget(self):
        if self.stop_requested():raise RunStopped()
        if time.time()>=self.deadline:raise BudgetExpired()

    def fit(self, replay, steps=None):
        losses=[]
        for _ in range(steps if steps is not None else self.cfg['model_updates']):
            self.check_budget()
            batch=replay.indices(self.cfg['batch'],self.cfg['model_horizon'],split='train')
            s=batch['s'][0];ctx=batch['ctx'][0];history=batch['history'][0]
            loss=0.
            for j in range(self.cfg['model_horizon']):
                # Context and command realizations are logged inputs. No future
                # physical observations are injected into this autoregression.
                s,history,_,_,_,_=advance(self.task,self.model,s,history,batch['ctx'][j],
                                         batch['command'][j],batch['scale'][j])
                loss=loss+self.model.error(s,batch['sn'][j])/self.cfg['model_horizon']
            if self.cfg.get('failure_fraction', 0.):
                one, weights, sampling = replay.failure_sample(self.cfg['batch'], self.cfg['failure_fraction'])
            else:
                one=replay.indices(self.cfg['batch'],split='train')
                weights=torch.ones(self.cfg['batch'],device=one['s'].device)
                sampling=dict(failure_batch_rate=float(one['failure'][0].float().mean()),failure_mix=0.)
            # Match the hand reference: termination labels train the classifier,
            # not the physical state predictor through a classification shortcut.
            with torch.no_grad():
                next_s,_=self.model(one['s'][0],one['command'][0],one['ctx'][0],one['scale'][0])
            logit=self.model.termination_logit(next_s,one['ctx'][0])
            term=(nn.functional.binary_cross_entropy_with_logits(
                logit,one['failure'][0].float(),reduction='none')*weights).mean()
            norm=checked_step(loss+term,self.model_opt,self.model.parameters())
            self.counts['model']+=1
            losses.append((float(loss),float(term),norm))
        return dict(model_loss=sum(v[0] for v in losses)/len(losses),
                    terminal_loss=sum(v[1] for v in losses)/len(losses),model_grad=losses[-1][2],**sampling)

    def imagine(self, replay):
        self.check_budget()
        b=replay.indices(self.cfg['batch'],split='train')
        s,h,ctx=b['s'][0],b['history'][0],b['ctx'][0]
        rewards,continuations,features,valid_steps=[],[],[],[]
        alive=torch.ones(len(s),dtype=torch.bool,device=s.device)
        with frozen(self.model,self.critic):
            for _ in range(self.cfg['horizon']):
                features.append(torch.cat((s,ctx),-1))
                valid_steps.append(alive)
                command=self.task.imagined_command(self.actor(h),ctx)
                scale=self.task.cfg['relative_scale']*(1+(torch.rand(len(s),1,device=s.device)-.5)*.1)
                s,h,r,logit,timeout,_=advance(self.task,self.model,s,h,ctx,command,scale)
                continuation=1-torch.sigmoid(logit)
                # Bootstrap exactly at time limits; later imagined actions have
                # zero weight. Failure probability suppresses timeout bootstrap.
                tail=self.critic(torch.cat((s,ctx),-1))
                rewards.append(r+self.cfg['gamma']*continuation*timeout.float()*tail)
                continuations.append(continuation*(~timeout).float())
                alive=alive&~timeout
            features.append(torch.cat((s,ctx),-1))
            J=objective(rewards,continuations,self.critic(features[-1]),self.cfg['gamma'])
            norm=checked_step(-J.mean(),self.actor_opt,self.actor.parameters(),.1)
            assert all(p.grad is None for p in self.model.parameters())
            assert all(p.grad is None for p in self.critic.parameters())
        self.counts['actor']+=1
        with torch.no_grad():
            x=torch.stack(features).detach()
            r=torch.stack(rewards).detach();c=torch.stack(continuations).detach()
            valid=torch.stack(valid_steps)
            # Real-only arms keep the statistics collected in Collector.step.
            # Do not let imagined states redefine the critic input distribution.
            if self.cfg.get('critic_norm_source', 'mixed') == 'mixed':
                self.critic.norm.update(x[:-1][valid])
            v=self.critic(x)
            target=lambda_returns(r,c,v,self.cfg['gamma'],self.cfg['lambda'])
        x=x[:-1][valid];target=target[valid]
        for _ in range(self.cfg['critic_epochs']):
            for ids in torch.randperm(len(x),device=x.device).chunk(4):
                self.check_budget()
                loss=(self.critic(x[ids])-target[ids]).square().mean()
                checked_step(loss,self.critic_opt,self.critic.parameters())
                self.counts['critic']+=1
        self.counts['cycles']+=1
        return dict(actor_objective=float(J.mean()),actor_grad=norm,critic_loss=float(loss),
                    imagined_reward=float(r.mean()),survival=float(c.prod(0).mean()))

    def ppo(self, rows):
        b={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
        adv,ret=gae(b['reward'],b['value'],b['next_value'],b['failure'],b['timeout'],b['valid'],
                    self.cfg['gamma'],self.cfg['lambda'])
        valid=b['valid'].flatten()
        flat={k:v.flatten(0,1)[valid] for k,v in b.items()}
        adv=adv.flatten()[valid];ret=ret.flatten()[valid]
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        x=torch.cat((flat['s'],flat['ctx']),-1)
        # PPO uses the same actor architecture/initial means. It learns its own
        # Gaussian scale; raw samples, not clipped commands, define likelihoods.
        kls=[]
        old_sigma=self.log_std.exp().detach().clone()
        for _ in range(4):
            epoch_kls=[]
            for ids in torch.randperm(len(adv),device=adv.device).split(self.cfg['ppo_batch']):
                self.check_budget()
                mean=self.actor(flat['history'][ids])
                dist=torch.distributions.Normal(mean,self.log_std.exp().expand_as(mean))
                logp=dist.log_prob(flat['raw'][ids]).sum(-1)
                loss=actor_loss(-flat['logp'][ids],-logp,adv[ids],True,.2).mean()
                norm=checked_step(loss,self.ppo_opt,list(self.actor.parameters())+[self.log_std])
                self.counts['actor']+=1
                val=self.critic(x[ids])
                vloss=critic_loss(flat['value'][ids],val,.2,ret[ids],True).mean()
                checked_step(vloss,self.critic_opt,self.critic.parameters())
                self.counts['critic']+=1
                old=torch.distributions.Normal(flat['mean'][ids],old_sigma.expand_as(mean))
                kl=float(torch.distributions.kl_divergence(old,dist).sum(-1).mean().detach())
                kls.append(kl);epoch_kls.append(kl)
            lr,_=self.ppo_scheduler.update(self.ppo_opt.param_groups[0]['lr'],0.,0,0,
                                            sum(epoch_kls)/len(epoch_kls))
            for group in self.ppo_opt.param_groups:group['lr']=lr
        self.counts['cycles']+=1
        # Update moments only after the whole on-policy batch has been consumed.
        self.actor.norm.update(flat['history']);self.critic.norm.update(x)
        return dict(actor_loss=float(loss),critic_loss=float(vloss),actor_grad=norm,
                    gaussian_kl=sum(kls)/len(kls),sigma=float(self.log_std.exp().mean()),
                    actor_lr=self.ppo_opt.param_groups[0]['lr'])

    @torch.no_grad()
    def validate_model(self,replay):
        b=replay.indices(self.cfg['batch'],self.cfg['model_horizon'],split='valid')
        s,h=b['s'][0],b['history'][0];errors=[]
        for j in range(self.cfg['model_horizon']):
            s,h,r,_,_,_=advance(self.task,self.model,s,h,b['ctx'][j],b['command'][j],b['scale'][j])
            errors.append(float(self.model.error(s,b['sn'][j])))
        # One-step unbiased validation, including terminal transitions. Metrics
        # never update parameters/statistics or select replay training samples.
        one=replay.indices(max(1024,self.cfg['batch']),split='valid')
        predicted,logit=self.model(one['s'][0],one['command'][0],one['ctx'][0],one['scale'][0])
        prob=logit.sigmoid();truth=one['failure'][0].float()
        return dict(heldout_model_h1_mse=errors[0],heldout_model_hk_mse=errors[-1],
                    heldout_failure_brier=float((prob-truth).square().mean()),
                    heldout_failure_rate=float(truth.mean()),heldout_predicted_failure_rate=float(prob.mean()),
                    heldout_failure_positives=int(truth.sum()),
                    heldout_failure_true_positive=int(((prob>=.5)&truth.bool()).sum()),
                    heldout_failure_false_positive=int(((prob>=.5)&~truth.bool()).sum()),
                    heldout_reward_hk_rmse=float((r-b['reward'][-1]).square().mean().sqrt()),
                    heldout_joint_hk_rmse=float((s[:,S['q']]-b['sn'][-1,:,S['q']]).square().mean().sqrt()))

    def state_dict(self):
        return dict(actor=self.actor.state_dict(),critic=self.critic.state_dict(),model=self.model.state_dict(),
                    log_std=self.log_std.detach(),counts=self.counts.copy(),
                    optimizers={k:getattr(self,k).state_dict() for k in
                                ['actor_opt','critic_opt','model_opt','ppo_opt']})

    def load_state_dict(self, saved):
        for k in ['actor','critic','model']:
            getattr(self,k).load_state_dict(saved[k],strict=True)
        with torch.no_grad():self.log_std.copy_(saved['log_std'])
        for k,v in saved['optimizers'].items():getattr(self,k).load_state_dict(v)
        self.counts=saved['counts'].copy()
