"""Native Sharpa collection and validity-aware use of the reference replay."""
import torch
from in_hand_rotation_mjlab.world_model.replay import Replay
from unionmimic.online_baseline.simulator import make_simulator, state, context
from unionmimic.online_baseline.task import S
from .model import pack, legacy_state, DX, DC, H


class SharpaReplay(Replay):
    def __init__(self,n,device):super().__init__(1000,n,DX,22,DC,73,0,2,device,d_plant=1)
    def failure_rows(self,batch):
        term=(self.done==1).nonzero();nt=min(len(term),batch//4)
        rr,ee,n=[],[],0
        for _ in range(30):
            r=torch.randint(min(self.G,self.L),(2*batch,),device=self.device)
            e=torch.randint(self.N,(2*batch,),device=self.device);ok=self.epi[r,e]>=0
            rr.append(r[ok]);ee.append(e[ok]);n+=int(ok.sum())
            if n>=batch-nt:break
        if n<batch-nt:raise RuntimeError('Not enough valid transitions for failure sampling')
        r=torch.cat(rr)[:batch-nt];e=torch.cat(ee)[:batch-nt]
        if nt:
            pick=term[torch.randint(len(term),(nt,),device=self.device)]
            r=torch.cat((r,pick[:,0]));e=torch.cat((e,pick[:,1]))
        return r,e


class SharpaCollector:
    def __init__(self,env,native,actor,replay):
        self.env,self.native,self.actor,self.replay=env,native,actor,replay
        n=env.num_envs;device=env.device
        self.frame=env.obs_buf[:,:73].clamp(-5,5).clone()
        self.history=self.frame[:,None].expand(-1,H,-1).clone()
        self.episode=torch.arange(n,device=device);self.next_episode=n
        self.age=torch.zeros(n,dtype=torch.long,device=device)
        self.ep_drops=0;self.valid_rows=0;self.max_error=dict(reward=0.,action=0.,target=0.)
        self.actor_updates=0;self.used_actor_updates=0;self.resets=0

    @torch.no_grad()
    def collect(self,noise):
        env=self.env;invalid=env.reset_buf.bool().clone();valid=~invalid
        if invalid.any():
            ids=invalid.nonzero().flatten();n=len(ids)
            self.episode[ids]=torch.arange(self.next_episode,self.next_episode+n,device=env.device)
            self.next_episode+=n;self.age[ids]=0;self.resets+=n
        raw=state(env);base=env.root_state_tensor[env.hand_indices,:13].clone()
        c=torch.cat((context(env),base[:,:7]),-1);s=pack(raw,base)
        if valid.any():self.actor.norm.update(self.history[valid].flatten(1))
        a=(self.actor(self.history.flatten(1))+noise*torch.randn(env.num_envs,22,device=env.device)).clamp(-1,1)
        env.step(a);raw_next=state(env);sn=pack(raw_next,base)
        # Independent native reward computation on converted physical/sensor state.
        r,failure,timeout,_,metrics=self.native.reward(legacy_state(s,c),legacy_state(sn,c),c[:,:93])
        action,target=self.native.controller(raw,env.study_command,env.relative_scale_tensor)
        if valid.any():
            for key,error in [('reward',(r-env.rew_buf).abs()),('action',(action-env.actions).abs()),('target',(target-env.prev_targets).abs())]:
                self.max_error[key]=max(self.max_error[key],float(error[valid].max()))
            assert self.max_error['reward']<=2e-3 and self.max_error['action']<=1e-6 and self.max_error['target']<=1e-6,self.max_error
            assert torch.equal((failure|timeout)[valid],env.reset_buf.bool()[valid]), 'Termination mismatch'
        for x in (s,sn,c,env.rew_buf,self.frame):assert torch.isfinite(x).all(),'Nonfinite simulator state'
        done=torch.where(env.reset_buf.bool(),torch.where(timeout,2,1),0).to(torch.uint8)
        done=torch.where(valid,done,torch.zeros_like(done))
        self.ep_drops+=int((done==1).sum());self.valid_rows+=int(valid.sum())
        self.replay.add(s=s,s_next=sn,a=env.study_command,ctx=c,frame=self.frame,
                        info_ok=valid,ep_step=self.age.clone(),epi=torch.where(valid,self.episode,-1),
                        plant=env.relative_scale_tensor,rew=torch.stack((env.rew_buf,metrics['rotation']),-1),done=done)
        self.used_actor_updates=max(self.used_actor_updates,self.actor_updates)
        self.frame=env.obs_buf[:,:73].clamp(-5,5).clone()
        self.history=torch.cat((self.history[:,1:],self.frame[:,None]),1)
        self.history[invalid]=self.frame[invalid,None].expand(-1,H,-1)
        self.age+=valid.long()
