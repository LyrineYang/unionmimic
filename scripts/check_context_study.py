"""Context integration checks independent of the simulator training loop."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import unionmimic.runtime
import argparse,json
import h5py,torch
from types import SimpleNamespace
from unionmimic.models import load_world_model,state_model_digest
from unionmimic.common import write_json,digest
from unionmimic.context_study import ContextActor,ContextReplay
from in_hand_rotation_mjlab.world_model.policy import Actor

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=3);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(47);dev=torch.device('cuda',a.gpu);torch.cuda.set_device(dev)
    source=ROOT/'work/analysis/sm-context-trajectories-20261109-v1'
    wm,ck,cfg=load_world_model(source/'config.yaml',gpu=a.gpu);del ck;wm.requires_grad_(False);before=digest(wm)
    with h5py.File(source/'episodes.h5') as f:
        assert f.attrs['state_model_sha256']==state_model_digest(wm)
        g=f['000002'];t=30
        latent=dict(deter=torch.tensor(g['post_deter'][t:t+1],device=dev),
                    stoch=torch.nn.functional.one_hot(torch.tensor(g['post_index'][t:t+1],device=dev).long(),32).float())
        action=torch.tensor(g['executed_action'][t+1:t+2],device=dev,requires_grad=True)
        expected=torch.tensor(g['post_deter'][t+1:t+2],device=dev)
    state=wm.dynamics.img_step(latent,action,sample=True)
    error=float((state['deter']-expected).abs().max());assert error<1e-5
    state['deter'].square().mean().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all() and action.grad.abs().sum()>0
    assert all(p.grad is None for p in wm.parameters()) and before==digest(wm)
    task=SimpleNamespace(context_state=None);replay=ContextReplay(1,dev,task)
    replay.G=40;replay.epi[:40]=0;replay.ep_step[:40,0]=torch.arange(40,device=dev);replay.info_ok[:40]=True
    for i in range(40):
        replay.deter[i]=latent['deter']+i/100
        replay.indices[i]=latent['stoch'].argmax(-1)
    replay.epi[20,0]=-1;replay.info_ok[20,0]=False;replay.ep_step[20:30,0]=torch.arange(10,device=dev)
    g,e=replay.sample_starts(64,10)
    assert not ((g>=20)&(g<30)).any()
    torch.testing.assert_close(task.context_state['deter'],replay.deter[g,e],atol=0,rtol=0)
    torch.testing.assert_close(task.context_state['stoch'],latent['stoch'].expand(len(g),-1,-1),atol=0,rtol=0)
    # Copy the zero-context actor's history weights into the native actor: exact output equivalence.
    ctx=ContextActor('none').to(dev);base=Actor(730,22).to(dev)
    base.norm.load_state_dict(ctx.norm.state_dict())
    with torch.no_grad():
        for i,(l1,l2) in enumerate(zip(ctx.net,base.net)):
            if isinstance(l1,torch.nn.Linear):
                l2.weight.copy_(l1.weight[:,:730] if i==0 else l1.weight);l2.bias.copy_(l1.bias)
        obs=torch.randn(5,730,device=dev);ctx.real_context=torch.randn(5,512,device=dev)
        torch.testing.assert_close(ctx(obs),base(obs),atol=1e-7,rtol=1e-5)
        old=ctx(obs);ctx.real_context=torch.randn(5,512,device=dev)*100
        torch.testing.assert_close(ctx(obs),old,atol=0,rtol=0)
    # Fixed context accepts history + start state and differentiates to both.
    fixed=ContextActor('fixed').to(dev);x=torch.randn(5,1242,device=dev,requires_grad=True);fixed(x).sum().backward()
    assert x.grad[:,:730].abs().sum()>0 and x.grad[:,730:].abs().sum()>0
    write_json(Path(a.out).resolve(),dict(passed=True,checks=['cached full RSSM state exact','reset-invalid starts excluded',
        'one-step recorded executed-action alignment','frozen prior permits input gradients','zero-context actor native equivalence',
        'zero-context insensitive to prior values','context actor input gradients'],one_step_max_error=error,
        action_gradient_norm=float(action.grad.norm()),prior_unchanged=True))
if __name__=='__main__':main()
