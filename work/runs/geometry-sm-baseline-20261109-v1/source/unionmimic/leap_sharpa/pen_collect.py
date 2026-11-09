"""Shared simulator collection for pen SM and PPO; explicit reset validity."""
import torch
from in_hand_rotation_mjlab.world_model.replay import Replay
from unionmimic.online_baseline.simulator import state,context
from .model import pack
from .pen_model import DX,DC,DF,H,P
from .simulator import SharpaReplay


def snapshot(env):
    base=env.root_state_tensor[env.hand_indices,:13].clone();s=pack(state(env),base)
    s[:,176]=env.reference_step
    cam=env.pose_sensor
    s=torch.cat((s,cam.previous,cam.held,cam.hold_ticks[:,None].float(),cam.ready[:,None].float(),
        env.raw_previous,env.raw_previous_previous,env.current_charge),-1).clone()
    c=torch.cat((context(env),base[:,:7],env.task_ids[:,None].float(),env.clip_ids[:,None].float(),cam.bias_pos,cam.bias_quat),-1).clone()
    assert s.shape[-1]==DX and c.shape[-1]==DC
    return s,c


class PenReplay(SharpaReplay):
    def __init__(self,n,device):Replay.__init__(self,1000,n,DX,22,DC,DF,0,2,device,d_plant=1)
    def sample_windows(self,B,k):
        lo,hi=max(0,self.G-self.L),self.G-k
        if hi<=lo:raise RuntimeError('Not enough reference transitions')
        gs=[];es=[];count=0
        for _ in range(100):
            g=torch.randint(lo,hi+1,(2*B,),device=self.device);e=torch.randint(self.N,(2*B,),device=self.device)
            j=torch.arange(k,device=self.device)[:,None];ep=self.epi[(g[None]+j)%self.L,e[None]]
            ok=(ep==ep[:1]).all(0)&(ep[0]>=0);gs.append(g[ok]);es.append(e[ok]);count+=int(ok.sum())
            if count>=B:return torch.cat(gs)[:B],torch.cat(es)[:B]
        raise RuntimeError('Insufficient same-episode windows: inspect physical initialization, not sample forever')


class PenCollector:
    def __init__(self,env,task,actor,replay=None,critic=None,ppo=None):
        self.env,self.task,self.actor,self.replay,self.critic,self.ppo=env,task,actor,replay,critic,ppo
        self.frame=env.policy_frame.clone();self.history=self.frame[:,None].expand(-1,H,-1).clone()
        self.episode=torch.arange(env.num_envs,device=env.device);self.next_episode=env.num_envs
        self.age=torch.zeros_like(self.episode);self.valid_rows=0;self.recollections=0;self.actor_updates=0;self.used_actor_updates=0
        self.max_error=dict(reward=0.,action=0.,target=0.);self.ep_drops=0
    @torch.no_grad()
    def collect(self,noise=.1):
        env=self.env;invalid=env.reset_buf.bool().clone();valid=~invalid
        if invalid.any():
            ids=invalid.nonzero().flatten();self.episode[ids]=torch.arange(self.next_episode,self.next_episode+len(ids),device=env.device)
            self.next_episode+=len(ids);self.age[ids]=0
        s,rawc=snapshot(env);c=self.task.model.normalize(rawc);obs=self.history.flatten(1).clone()
        if self.ppo is None:
            if valid.any():self.actor.norm.update(obs[valid])
            raw=self.actor(obs)+noise*torch.randn(env.num_envs,22,device=env.device)
        else:
            dist=self.ppo.distribution(obs);raw=dist.sample();x=self.task.critic_features(s,None,None,c);value=self.critic(x)
        reference=env.pen_tasks.target(env.task_ids,env.clip_ids,env.reference_step)['q']
        env.step(raw.clamp(-1,1));sn,_=snapshot(env)
        reward,failure,timeout,metrics=self.task.reward(s,sn,c)
        target,residual=env.pen_controller(reference,env.study_command,s[:,148:170])
        if valid.any():
            for key,error in [('reward',(reward-env.rew_buf).abs()),('action',(residual-env.last_actions).abs()),('target',(target-env.prev_targets).abs())]:
                self.max_error[key]=max(self.max_error[key],float(error[valid].max()))
            assert self.max_error['reward']<2e-5 and self.max_error['action']<1e-6 and self.max_error['target']<1e-6,self.max_error
            assert torch.equal((failure|timeout)[valid],env.reset_buf.bool()[valid]),'Reference termination mismatch'
        for t in (s,sn,rawc,reward,self.frame):assert torch.isfinite(t).all(),'Nonfinite reference transition'
        done=torch.where(failure,1,torch.where(timeout,2,0)).to(torch.uint8)
        done=torch.where(valid,done,torch.zeros_like(done));self.ep_drops+=int((done==1).sum());self.valid_rows+=int(valid.sum())
        if self.replay is not None:
            self.replay.add(s=s,s_next=sn,a=env.study_command,ctx=rawc,frame=self.frame,info_ok=valid,
                ep_step=self.age.clone(),epi=torch.where(valid,self.episode,-1),plant=env.relative_scale_tensor,
                rew=torch.stack((env.rew_buf,metrics['position_error_m']),-1),done=done)
        row=None
        if self.ppo is not None:
            nx=self.task.critic_features(sn,None,None,c)
            row=dict(history=obs,features=x,raw_action=raw,mean=dist.loc.clone(),std=dist.scale.clone(),logp=dist.log_prob(raw).sum(-1),
                value=value,next_value=self.critic(nx),reward=env.rew_buf.clone(),failure=failure,timeout=timeout&~failure,valid=valid)
        self.frame=env.policy_frame.clone();self.history=torch.cat((self.history[:,1:],self.frame[:,None]),1)
        self.history[invalid]=self.frame[invalid,None].expand(-1,H,-1);self.age+=valid.long()
        self.used_actor_updates=max(self.used_actor_updates,self.actor_updates)
        return row
