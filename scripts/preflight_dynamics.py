"""Run supervised RSSM loss/backward on real data, with zero updates."""
import argparse
import time
from pathlib import Path
import unionmimic.runtime
import torch
from unionmimic.models import load_world_model
from unionmimic.dynamics_fit import DynamicsFit,EpisodeChunks
from unionmimic.common import digest,write_json,file_digest


def main(args):
    wm,saved,cfg=load_world_model(args.dataset/'config.yaml',args.checkpoint,args.gpu)
    device=next(wm.parameters()).device;before=digest(wm)
    fit=DynamicsFit(wm,saved);failure_before=digest(fit.heads.failure)
    chunks=EpisodeChunks(args.dataset,digest(wm.encoder),length=32,seed=args.seed)
    reports=[]
    for seed in (args.seed,args.seed+1):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rows=chunks.sample(args.batch,device);started=time.monotonic()
        fit.heads.zero_grad(set_to_none=True)
        loss,metrics=fit.loss(rows)
        loss.backward();torch.cuda.synchronize(device)
        norms={}
        for name,module in [('rssm',wm.dynamics),('reward',wm.heads['reward']),('prop',wm.heads['decoder']._mlp),('failure',fit.heads.failure)]:
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            norms[name]=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
            assert norms[name]>0
        assert all(p.grad is None for p in wm.encoder.parameters())
        assert all(p.grad is None for p in wm.heads['decoder']._cnn.parameters())
        reports.append(dict(seed=seed,loss=float(loss),metrics=metrics,grad_norms=norms,seconds=time.monotonic()-started))
    assert before==digest(wm) and failure_before==digest(fit.heads.failure)
    chunks.close()
    write_json(args.output,dict(passed=True,optimizer_steps=0,weights_unchanged=True,encoder_and_image_decoder_frozen=True,
        checkpoint_sha256=file_digest(args.checkpoint),batch=args.batch,chunk_length=32,measurements=reports,
        pytorch_peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/1024**2,
        note='Gradient readiness only; fitting quality, cache rebuild after actual updates and policy preservation remain unmeasured.'))
    print(reports,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--seed',type=int,default=63001)
    main(p.parse_args())
