"""Exercise unchanged LEAP F/StateError/Replay on real Sharpa transitions.

Only a physical-model interface check: no reward/contact head or policy update.
Run with scripts/leap_python.sh, independently of the Isaac Gym process.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unionmimic.common import write_json, file_digest
from unionmimic.sharpa_leap import L, SharpaCommandMap
from in_hand_rotation_mjlab.world_model.model import WorldModel, WMConfig, StateError
from in_hand_rotation_mjlab.world_model.record import state_slices
from in_hand_rotation_mjlab.world_model.replay import Replay


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    run=Path(a.run).resolve()
    assert json.loads((run/'acceptance.json').read_text())['interface_passed']
    out=run/'reference-model'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    started=time.time()
    torch.manual_seed(0);torch.set_num_threads(4)
    for name,sl in state_slices(22).items():assert L[name]==sl,(name,L[name],sl)
    saved=torch.load(run/'transitions.pt',map_location='cpu',weights_only=False)
    cfg=saved['task_constants'];cpu=saved['data']
    assert cpu['s'].shape==(64,4096,79)
    dev='cuda:0';data={k:v.to(dev) for k,v in cpu.items()}
    replay=Replay(1000,4096,79,22,data['ctx'].shape[-1],73,0,1,dev)
    for t in range(64):
        dn=torch.where(data['done'][t],torch.where(data['timeout'][t],2,1),0).to(torch.uint8)
        epi=torch.where(data['valid'][t],data['episode'][t],-1)
        replay.add(s=data['s'][t],s_next=data['sn'][t],a=data['command'][t],ctx=data['ctx'][t],
                   frame=data['frame'][t],done=dn,epi=epi,rew=data['reward'][t,:,None])
    wm=WorldModel(WMConfig(nj=22,dc=data['ctx'].shape[-1],dt=cfg['control_dt'])).to(dev)
    error=StateError(22).to(dev)
    controller=SharpaCommandMap(cfg['lower'],cfg['upper']).to(dev)
    valid=data['valid'];s=data['s'][valid];sn=data['sn'][valid];ctx=data['ctx'][valid]
    mu=ctx.mean(0);sd=ctx.std(0).clamp_min(1e-6)
    wm.fit_stats(s,sn[:,L['cmd']],(ctx-mu)/sd,sn);error.fit(s,sn)
    optimizer=torch.optim.Adam(wm.parameters(),lr=2e-4)
    losses=[];max_target_error=0.
    before=wm.net[-1].weight.detach().clone()
    for update in range(10):
        g,e=replay.sample_windows(2048,8)
        s0,acts,truth,raw_ctx,done=replay.window(g,e,8)
        j=torch.arange(8,device=dev)[:,None];rr=g[None]+j
        assert data['valid'][rr,e[None]].all(), 'Native replay included reset-input frame'
        # Explicitly retain action filter memory and actual recorded scale.
        sh=s0;memory=data['last_action'][g,e];loss=0.
        for step in range(8):
            cmd,memory=controller(sh[:,L['cmd']],acts[step],memory,data['scale'][g+step,e])
            max_target_error=max(max_target_error,float((cmd-truth[step,:,L['cmd']]).abs().max().detach()))
            sh,_=wm(sh,cmd,(raw_ctx-mu)/sd)
            loss=loss+error(sh,truth[step]).mean()/8
        optimizer.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(wm.parameters(),1.0);optimizer.step()
        losses.append(float(loss.detach()))
    assert max_target_error<=1e-6
    assert not torch.equal(before,wm.net[-1].weight), 'Reference physical model did not update'
    for p in wm.parameters():p.requires_grad_(False)
    command=acts[0].detach().clone().requires_grad_(True)
    cmd,_=controller(s0[:,L['cmd']],command,data['last_action'][g,e],data['scale'][g,e])
    prediction,_=wm(s0,cmd,(raw_ctx-mu)/sd)
    grad=torch.autograd.grad(error(prediction,truth[0]).mean(),command)[0]
    assert torch.isfinite(grad).all() and grad.abs().max()>0, 'No finite command-to-physics gradient'
    torch.save(dict(wm=wm.state_dict(),config=wm.cfg.__dict__,purpose='interface diagnostic only'),out/'diagnostic-model.pt')
    result=dict(physical_model_interface_passed=True,full_sm_loop_passed=False,
        source_commit='0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0',nj=22,
        hidden=list(wm.cfg.hidden),batch=2048,window=8,replay_capacity=1000,envs=4096,
        physical_model_updates=10,actor_updates=0,critic_updates=0,losses=losses,
        max_target_error=max_target_error,input_gradient_abs_max=float(grad.abs().max()),
        cross_reset_windows=False,seconds=time.time()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        limitations=['Physical-model loss only; termination/contact/reward supervision not integrated',
                     'No online transport, imagined observation or policy BPTT acceptance'],
        dataset_sha256=file_digest(run/'transitions.pt'))
    write_json(out/'acceptance.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
