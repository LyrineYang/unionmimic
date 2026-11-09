"""Frozen-state versus action-prior evolution on historical held-out episodes."""
import argparse,json
from pathlib import Path
import unionmimic.runtime
import torch,h5py,numpy as np
from unionmimic.models import load_world_model,state_model_digest
from unionmimic.common import write_json

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.dataset).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    wm,saved,cfg=load_world_model(root/'config.yaml',gpu=a.gpu);del saved;wm.requires_grad_(False);dev=next(wm.parameters()).device;torch.set_num_threads(4)
    blocks=[];episodes=[]
    with h5py.File(root/'episodes.h5') as f:
        assert f.attrs['complete'] and f.attrs['state_model_sha256']==state_model_digest(wm)
        for key,g in f.items():
            if g.attrs['split']!='valid' or len(g['prop'])<27:continue
            ix=np.unique(np.linspace(9,len(g['prop'])-17,min(64,len(g['prop'])-26)).astype(int));future=ix[:,None]+np.arange(1,17)[None]
            blocks.append(dict(deter=g['post_deter'][:][ix],logit=g['post_logit'][:][ix],index=g['post_index'][:][ix],
                action=g['executed_action'][:][future],future_deter=g['post_deter'][:][future],prop=g['prop'][:][future],start_prop=g['prop'][:][ix]))
            episodes.append(key)
    data={k:torch.as_tensor(np.concatenate([b[k] for b in blocks]),device=dev) for k in blocks[0]}
    initial={k:data[k] for k in ('deter','logit')};initial['stoch']=torch.nn.functional.one_hot(data['index'].long(),wm.dynamics._discrete).float()
    rows=[]
    for mode in ('fixed_context','action_prior_mode','action_prior_sample'):
        torch.manual_seed(117);state={k:v.clone() for k,v in initial.items()}
        for t in range(16):
            if mode!='fixed_context':state=wm.dynamics.img_step(state,data['action'][:,t],sample=mode.endswith('sample'))
            if t+1 not in (1,2,4,8,16):continue
            prediction=wm.heads['decoder']._mlp(wm.dynamics.get_feat(state))['prop'].mode()
            truth=data['prop'][:,t];latent_error=float((state['deter']-data['future_deter'][:,t]).square().mean().sqrt())
            rows.append(dict(mode=mode,horizon=t+1,latent_rmse=latent_error,
                joint_rmse=float((prediction[:,:22]-truth[:,:22]).square().mean().sqrt()),
                tactile_brier=float((prediction[:,44:49].clamp(0,1)-truth[:,44:49]).square().mean()),
                observation_persistence_joint_rmse=float((data['start_prop'][:,:22]-truth[:,:22]).square().mean().sqrt()),
                observation_persistence_tactile_brier=float((data['start_prop'][:,44:49]-truth[:,44:49]).square().mean())))
            if t==0 and mode!='fixed_context':assert latent_error<2e-4,'Incoming-action/state alignment failed'
    write_json(out/'result.json',dict(rows=rows,episodes=episodes,starts=len(data['deter']),control_dt=.10002,
        source=str(root),state_model_sha256=state_model_digest(wm),weights_frozen=True,future_inputs='recorded executed actions only',
        limits='Held-out historical 10Hz block task. Latent labels are teacher future posteriors and tactile labels are noisy sensor bits. Decoders are used for offline diagnostics only. Not a 50Hz pen bridge or actor-training result.'))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
