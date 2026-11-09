# SPDX-License-Identifier: Apache-2.0
# Adapted from Wuji Dex-Supermimic 2becb738 pen_spin/mdp/commands.py.
"""Reference-state sampler with outcome accounting for every traversed bin.

The only task adaptation is conditioning the clip pool on R1/R2, so the four
tasks retain equal environment allocation. No outcome changes known targets.
"""
import torch

class ReferenceSampler:
    def __init__(self,lengths):
        self.lengths=lengths;self.device=lengths.device;self.bin_size=25
        slots=(lengths-1).clamp_min(1);self.counts=(slots+24)//25
        self.offsets=torch.cat((torch.zeros(1,device=self.device,dtype=torch.long),self.counts.cumsum(0)[:-1]))
        self.clips=torch.repeat_interleave(torch.arange(len(lengths),device=self.device),self.counts)
        self.starts=(torch.arange(len(self.clips),device=self.device)-self.offsets[self.clips])*25
        self.stops=torch.minimum(self.starts+25,slots[self.clips]);self.scores=torch.zeros(len(self.clips),device=self.device)
        self.pending=torch.zeros(2,len(self.clips),device=self.device);self.steps=0
    def bins(self,clip,phase):
        phase=phase.clamp_min(0).minimum((self.lengths[clip]-2).clamp_min(0))
        return self.offsets[clip]+torch.div(phase,self.bin_size,rounding_mode='floor').long().minimum(self.counts[clip]-1)
    def record(self,clip,first,last,failed):
        if not len(clip):return
        first=self.bins(clip,first);last=self.bins(clip,last);n=len(self.clips)
        assert (last>=first).all()
        entered=torch.bincount(first,minlength=n+1)-torch.bincount(last+1,minlength=n+1)
        self.pending[0]+=entered.cumsum(0)[:n]
        # Outcomes are Boolean, so integer counting is exactly equivalent to
        # weighted bincount and supports deterministic CUDA in Torch 2.1.
        self.pending[1]+=torch.bincount(last[failed.bool()],minlength=n)
    def update(self):
        self.steps+=1
        if self.steps<40:return
        self.steps=0;count,fail=self.pending;mask=count>0;decay=.99**count[mask]
        self.scores[mask]=decay*self.scores[mask]+(1-decay)*fail[mask]/count[mask];self.pending.zero_()
    def probabilities(self,pool):
        mask=(self.clips[:,None]==pool[None]).any(-1);indices=mask.nonzero().flatten();scores=self.scores[indices]
        uniform=torch.full_like(scores,1/len(scores));proportional=torch.where(scores.sum()>0,scores/scores.sum().clamp_min(torch.finfo(scores.dtype).eps),uniform)
        return indices,.2*uniform+.8*proportional
    def sample(self,pool,n):
        # Torch 2.1 CUDA multinomial internally uses nondeterministic float
        # cumsum. CPU categorical sampling preserves the same probabilities
        # while keeping strict deterministic mode; transfer only bin indices.
        indices,p=self.probabilities(pool)
        draw=torch.multinomial(p.cpu(),n,replacement=True).to(self.device);bins=indices[draw]
        steps=self.starts[bins]+(torch.rand(n,device=self.device)*(self.stops[bins]-self.starts[bins])).long()
        held=torch.rand(n,device=self.device)<.15
        steps=torch.where(held,-torch.randint(25,151,(n,),device=self.device),steps)
        return self.clips[bins],steps
    def state_dict(self):return dict(scores=self.scores.clone(),pending=self.pending.clone(),steps=self.steps)
    def load_state_dict(self,d):
        self.scores.copy_(d['scores']);self.pending.copy_(d['pending']);self.steps=int(d['steps'])
