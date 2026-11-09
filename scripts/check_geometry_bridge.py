"""Replay-control audit of compact geometry vs real estimates; never actor supervision."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import argparse,json,time
import torch
from unionmimic.leap_sharpa.model import SharpaModel
from unionmimic.geometry_context import PositionErrorBridge
from unionmimic.common import write_json,file_digest,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.cuda.set_device(a.gpu);dev=torch.device('cuda',a.gpu)
    datapath=ROOT/'work/demos/geometry-live-20261109-v1/transitions.pt';data={k:v.to(dev) for k,v in torch.load(datapath,map_location='cpu').items()}
    ckpath=ROOT/'work/runs/sharpa-leap-12h-20261103-v1/ckpt/best.pt';ck=torch.load(ckpath,map_location='cpu')
    model=SharpaModel(ck['task_constants']).to(dev);model.load_state_dict(ck['wm']);model.eval().requires_grad_(False)
    bridgepath=ROOT/'work/analysis/geometry-readout-20261109-v1/bridge.pt'
    bridge=PositionErrorBridge.from_checkpoint(torch.load(bridgepath,map_location='cpu')).to(dev);before=digest(model)
    torch.manual_seed(491);starts=[]
    for g in range(10,len(data['state'])-16):
        ok=(data['episode'][g:g+16]==data['episode'][g]).all(0)&(data['episode'][g]>=0)
        starts.extend((g,int(e)) for e in ok.nonzero().flatten().cpu())
    ids=torch.randperm(len(starts))[:512];pairs=torch.tensor(starts,device=dev)[ids];g,e=pairs[:,0],pairs[:,1]
    s=data['state'][g,e];c=model.normalize(data['context'][g,e]);initial_error=data['estimate'][g,e]-s[:,66:69]
    error=initial_error.clone();rows=[]
    with torch.no_grad():
        for h in range(16):
            s,_=model(s,data['command'][g+h,e],c,scale=data['scale'][g+h,e])
            pred,error=bridge.step(s[:,66:69],error,torch.zeros_like(error))
            if h==0:
                target_error=float((s[:,44:66]-data['next_state'][g,e,44:66]).abs().max());assert target_error<1e-6
            if h+1 in (1,4,8,16):
                real=data['next_estimate'][g+h,e];truth=data['next_state'][g+h,e,66:69]
                rows.append(dict(horizon=h+1,physical_position_error_mm=float((s[:,66:69]-truth).norm(dim=-1).mean()*1000),
                    raw_physics_to_real_estimate_mm=float((s[:,66:69]-real).norm(dim=-1).mean()*1000),
                    fixed_error_to_real_estimate_mm=float((s[:,66:69]+initial_error-real).norm(dim=-1).mean()*1000),
                    ar1_mean_to_real_estimate_mm=float((pred-real).norm(dim=-1).mean()*1000),
                    sensor_only_ar1_error_mm=float((truth+error-real).norm(dim=-1).mean()*1000)))
    # Demonstrate the actual F -> geometry -> scalar policy-objective input path.
    s=data['state'][g[:64],e[:64]];c=model.normalize(data['context'][g[:64],e[:64]])
    command=torch.zeros(64,22,device=dev,requires_grad=True);sn,_=model(s,command,c,scale=data['scale'][g[:64],e[:64]])
    context,_=bridge.step(sn[:,66:69],initial_error[:64],torch.zeros_like(initial_error[:64]));context.square().sum().backward()
    assert torch.isfinite(command.grad).all() and command.grad.abs().sum()>0
    assert digest(model)==before and all(p.grad is None for p in model.parameters())
    write_json(out/'result.json',dict(rows=rows,starts=len(g),weights_unchanged=True,input_gradient_norm=float(command.grad.norm()),
        controller_target_max_error=target_error,data_sha256=file_digest(datapath),model_sha256=file_digest(ckpath),bridge_sha256=file_digest(bridgepath),
        protocol='Frozen historical F, complete-episode starts, recorded executed commands and controller scales; no future observations in F/bridge propagation.',
        limits='Sensor-only column substitutes ground-truth positions solely for offline error decomposition. It is not a usable actor input or a control comparison. Historical F/behavior are not the new random-policy baseline. AR1 scores use conditional mean, not stochastic coverage.'))
if __name__=='__main__':main()
