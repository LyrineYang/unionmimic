"""Frozen released WSM cost/gradient probe; no simulator or learner updates."""
import argparse,gc,json,time
from pathlib import Path
import unionmimic.runtime  # Isaac Gym before Torch; does not construct a simulator.
import torch,h5py
from unionmimic.models import load_world_model
from unionmimic.common import write_json,file_digest,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.dataset).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20261109)
    wm,saved,cfg=load_world_model(root/'config.yaml',gpu=a.gpu);del saved
    wm.requires_grad_(False);dev=next(wm.parameters()).device;before=digest(wm)
    counts={k:sum(p.numel() for p in m.parameters()) for k,m in [('encoder',wm.encoder),('rssm',wm.dynamics),('heads',wm.heads)]}
    with h5py.File(root/'episodes.h5') as f:
        assert f.attrs['complete'];g=next(iter(f.values()))
        prop=torch.tensor(g['prop'][:256],device=dev);depth=torch.tensor(g['depth'][:256],device=dev)
    results=[]
    def measure(label,fn):
        torch.cuda.synchronize(dev);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(dev)
        allocated=torch.cuda.memory_allocated(dev);start=time.perf_counter()
        try:
            info=fn();torch.cuda.synchronize(dev)
            row=dict(label=label,passed=True,seconds=time.perf_counter()-start,
                incremental_peak_gib=(torch.cuda.max_memory_allocated(dev)-allocated)/1024**3,**info)
        except torch.cuda.OutOfMemoryError as exc:
            row=dict(label=label,passed=False,error='CUDA OOM',seconds=time.perf_counter()-start)
            gc.collect();torch.cuda.empty_cache()
        results.append(row);write_json(out/'progress.json',results);print(json.dumps(row),flush=True)
    @torch.no_grad()
    def encode():
        for _ in range(4):embed=wm.encoder(dict(prop=prop,image=depth))
        return dict(batch=len(prop),repeats=4,embedding_dim=embed.shape[-1],decoders_called=False)
    with torch.no_grad():
        wm.encoder(dict(prop=prop,image=depth))
    torch.cuda.synchronize(dev)
    measure('real_encoder_no_grad',encode)
    for batch in (256,1024,4096):
        for horizon in (4,16):
            def run(batch=batch,horizon=horizon):
                state=wm.dynamics.initial(batch)
                actions=(torch.randn(horizon,batch,22,device=dev)*.1).requires_grad_()
                loss=0.
                for t in range(horizon):
                    state=wm.dynamics.img_step(state,actions[t],sample=True)
                    loss=loss+state['deter'].square().mean()/horizon
                loss.backward()
                assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum()>0
                assert all(p.grad is None for p in wm.parameters())
                return dict(batch=batch,horizon=horizon,input_gradient_norm=float(actions.grad.norm()),weight_gradients=False,decoders_called=False,sampling='released straight-through categorical sample')
            measure('rssm_input_bptt',run)
    assert before==digest(wm)
    write_json(out/'result.json',dict(parameters=counts,parameter_total=sum(counts.values()),weights_unchanged=True,
        measurements=results,recorded_control_dt=cfg['task']['sim']['dt']*cfg['task']['env']['controlFrequencyInv'],
        checkpoint_sha256=file_digest(Path(__file__).resolve().parents[1]/'work/checkpoints/wm_craftnet_set_z.pth'),
        limitation='Isolated FP32 context branch; excludes simulator, SM model, actor, critic, posterior replay and bridge. Synthetic initial states/actions for cost only; no control benefit demonstrated. Encoder is warmed up before timing; RSSM timings include state/action setup.'))
if __name__=='__main__':main()
