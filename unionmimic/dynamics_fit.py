"""Supervised RSSM adaptation using real episode histories and prior rollouts.

Uses WM-Craftnet's RSSM/heads/KL implementation. The multi-step objective and
feature-drift regularizer are this project's adaptation, not official SM code.
Encoder and image decoder stay frozen; RSSM and task prediction heads can learn.
"""
import copy
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from .models import CalibratedHeads, state_model_digest
from .common import digest


class EpisodeChunks:
    def __init__(self, dataset, encoder_sha256, length=32, seed=63001):
        self.dataset=Path(dataset);self.length=length;self.rng=np.random.default_rng(seed)
        self.file=h5py.File(self.dataset/'episodes.h5','r')
        assert self.file.attrs['complete']
        assert str(self.file.attrs['state_model_sha256']).split(':')[0]==encoder_sha256, 'Re-encode raw observations after changing encoder'
        self.pools={False:[],True:[]}
        for key,g in self.file.items():
            if g.attrs['split']=='train' and len(g['reward'])>=2:
                self.pools[bool(g['failure'][-1])].append(key)
        assert all(self.pools.values()), 'Need both successful and failure episodes in training'

    def sample(self,batch_size,device):
        rows=[]
        for i in range(batch_size):
            failed=bool(i%2);pool=self.pools[failed];key=pool[int(self.rng.integers(len(pool)))];g=self.file[key]
            n=len(g['reward'])
            offset=max(0,n-self.length) if failed else int(self.rng.integers(max(1,n-self.length+1)))
            end=min(n,offset+self.length);idx=np.minimum(np.arange(offset,offset+self.length),n-1)
            row={k:torch.tensor(g[k][:][idx],device=device) for k in ('embed','executed_action','prop','reward','failure')}
            row['valid']=torch.arange(self.length,device=device)<end-offset
            row['is_first']=torch.zeros(self.length,device=device);row['is_first'][0]=float(offset==0)
            row['prefix_embed']=torch.tensor(g['embed'][:offset],device=device)
            row['prefix_action']=torch.tensor(g['executed_action'][:offset],device=device)
            row['episode_id']=key;row['offset']=offset;rows.append(row)
        return rows

    def close(self):self.file.close()


@torch.no_grad()
def prefix_state(dynamics,rows):
    # Replay independent prefixes together. Inactive padded lanes retain their
    # previous state; none of the future chunk observations enter this state.
    device=rows[0]['embed'].device
    lengths=torch.tensor([len(r['prefix_embed']) for r in rows],device=device)
    state=dynamics.initial(len(rows))
    for t in range(int(lengths.max())):
        embed=torch.stack([r['prefix_embed'][t] if t<len(r['prefix_embed']) else torch.zeros_like(r['embed'][0]) for r in rows])
        action=torch.stack([r['prefix_action'][t] if t<len(r['prefix_action']) else torch.zeros_like(r['executed_action'][0]) for r in rows])
        proposed,_=dynamics.obs_step(state,action,embed,torch.full((len(rows),),float(t==0),device=device))
        state={k:torch.where((lengths>t).reshape(-1,*([1]*(v.ndim-1))),v,state[k]) for k,v in proposed.items()}
    return state


class DynamicsFit:
    def __init__(self,wm,saved):
        self.wm=wm;device=next(wm.parameters()).device
        self.heads=CalibratedHeads(wm).to(device)
        self.heads.failure.load_state_dict(saved['failure_head'])
        self.heads.failure_log_weight.fill_(saved['failure_log_weight'])
        self.reference=copy.deepcopy(wm.dynamics).requires_grad_(False).eval()
        self.heads.requires_grad_(False)
        wm.dynamics.requires_grad_(True);wm.heads['reward'].requires_grad_(True)
        wm.heads['decoder']._mlp.requires_grad_(True);self.heads.failure.requires_grad_(True)
        self.heads.eval();self.original_state_hash=state_model_digest(wm)

    def parameters(self):return [p for p in self.heads.parameters() if p.requires_grad]

    def loss(self,rows,horizon=4,drift_coefficient=.01):
        data={k:torch.stack([r[k] for r in rows]) for k in ('embed','executed_action','prop','reward','failure','valid','is_first')}
        # Pair reference/current categorical draws for the drift diagnostic.
        rng=torch.cuda.get_rng_state(data['embed'].device)
        with torch.no_grad():
            reference_start=prefix_state(self.reference,rows)
            reference,_=self.reference.observe(data['embed'],data['executed_action'].clone(),data['is_first'],reference_start)
        torch.cuda.set_rng_state(rng,data['embed'].device)
        start=prefix_state(self.wm.dynamics,rows)
        post,prior=self.wm.dynamics.observe(data['embed'],data['executed_action'].clone(),data['is_first'],start)
        kl,_,_,_=self.wm.dynamics.kl_loss(post,prior,1.,.5,.1)
        mask=data['valid'] & ~data['is_first'].bool()
        kl_loss=kl[mask].mean()
        drift=(post['deter']-reference['deter']).square().mean(-1)[data['valid']].mean()
        batch,length=data['valid'].shape
        state={k:v[:,:-1].reshape(batch*(length-1),*v.shape[2:]) for k,v in post.items()}
        rewards=[];props=[];failures=[]
        for h in range(1,horizon+1):
            future=(torch.arange(length-1,device=data['embed'].device)+h).clamp(max=length-1)
            action=data['executed_action'][:,future].reshape(-1,22)
            state=self.wm.dynamics.img_step(state,action)
            rd,pd,fl=self.heads.predictions(self.wm.dynamics.get_feat(state))
            valid=data['valid'][:,:-1] & data['valid'][:,future] & (torch.arange(length-1,device=action.device)[None]+h<length)
            valid=valid.reshape(-1)
            if not valid.any():
                continue
            target=data['prop'][:,future].reshape(-1,73);pred=pd.mode()
            prop=10*(pred[:,:22]-target[:,:22]).square().mean(-1)+2*(pred[:,44:49]-target[:,44:49]).square().mean(-1)+.1*(pred-target).square().mean(-1)
            rewards.append(-rd.log_prob(data['reward'][:,future].reshape(-1))[valid].mean())
            props.append(prop[valid].mean())
            failure=F.binary_cross_entropy_with_logits(fl+self.heads.failure_log_weight,
                data['failure'][:,future].reshape(-1).float(),reduction='none',
                pos_weight=self.heads.failure_log_weight.exp())
            failures.append(failure[valid].mean())
        if not rewards or not mask.any():
            raise ValueError('No valid transitions for dynamics supervision in this batch')
        reward_loss=torch.stack(rewards).mean();prop_loss=torch.stack(props).mean();failure_loss=torch.stack(failures).mean()
        loss=reward_loss+prop_loss+failure_loss+kl_loss+drift_coefficient*drift
        assert torch.isfinite(loss)
        return loss,dict(reward_loss=float(reward_loss),prop_loss=float(prop_loss),failure_loss=float(failure_loss),
                         kl_loss=float(kl_loss),feature_drift=float(drift),horizon=horizon,
                         supervised_transitions=int(mask.sum()),includes_early_failure=any(len(r['prefix_embed'])==0 and r['failure'].any() for r in rows))

    def checkpoint(self,saved,optimizer,step):
        result={**saved,'world_model':{'world_model':self.wm.state_dict()},
                'failure_head':self.heads.failure.state_dict(),'failure_log_weight':float(self.heads.failure_log_weight),
                'dynamics_state':dict(step=step,optimizer=optimizer.state_dict(),state_model_sha256=state_model_digest(self.wm)),
                'calibration_report':dict(passed=False,reason='Dynamics changed: rebuild posterior/prior cache and revalidate before actor updates')}
        # Preserve historical files; a new model needs a new critic/optimizer
        # initialization decision, not an implicit stale-state resume.
        for key in ('sm_state','sm_report','calibration_optimizer','calibration_rng'):
            result.pop(key,None)
        return result
