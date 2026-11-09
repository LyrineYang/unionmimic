"""Episode-held-out information screen on historical aligned trajectories.

Predict noisy future tactile observations, not privileged physical contact.
Only the small probe learns; WSM and historical control policies are unchanged.
"""
import argparse,json,time
from pathlib import Path
import torch,h5py,numpy as np
from unionmimic.common import write_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.dataset).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(a.gpu);dev=torch.device('cuda',a.gpu);start=time.time()
    splits={'train':[],'valid':[]};episodes={'train':[],'valid':[]}
    with h5py.File(root/'episodes.h5') as f:
        assert f.attrs['complete'];state_hash=str(f.attrs['state_model_sha256']);behavior=str(f.attrs.get('behavior_policy','released-policy/stress'))
        for key,g in f.items():
            n=len(g['prop']);ix=np.arange(9,n-4,2)
            if not len(ix):continue
            prop=g['prop'][:];history=np.stack([prop[ix-j] for j in range(10)],1).reshape(len(ix),730)
            splits[g.attrs['split']].append(dict(history=history,h=g['post_deter'][:][ix],embed=g['embed'][:][ix],
                target=prop[ix+4,44:49],episode=np.full(len(ix),int(key))))
            episodes[g.attrs['split']].append(key)
    assert not set(episodes['train'])&set(episodes['valid'])
    arrays={s:{k:torch.tensor(np.concatenate([b[k] for b in bs]),device=dev,dtype=torch.float32) for k in bs[0]} for s,bs in splits.items()}
    torch.manual_seed(113);proj_h=torch.randn(512,64,device=dev)/512**.5;proj_e=torch.randn(arrays['train']['embed'].shape[-1],64,device=dev)/arrays['train']['embed'].shape[-1]**.5
    # Identical readout width/parameter count; same initial readout and minibatches.
    variants=['history_only','history_plus_current_encoder','history_plus_wsm','history_plus_shuffled_wsm'];results=[]
    for variant in variants:
        features={}
        for s,rows in arrays.items():
            if variant=='history_only':extra=torch.zeros(len(rows['h']),64,device=dev)
            elif variant=='history_plus_current_encoder':extra=rows['embed']@proj_e
            else:
                extra=rows['h']@proj_h
                if variant.endswith('shuffled_wsm'):
                    generator=torch.Generator(device=dev).manual_seed(112 if s=='train' else 114)
                    extra=extra[torch.randperm(len(extra),device=dev,generator=generator)]
            features[s]=torch.cat((rows['history'],extra),-1)
        mu=features['train'].mean(0);sd=features['train'].std(0).clamp_min(.01)
        features={s:((x-mu)/sd).clamp(-10,10) for s,x in features.items()}
        torch.manual_seed(115);model=torch.nn.Sequential(torch.nn.Linear(794,256),torch.nn.ELU(),torch.nn.Linear(256,5)).to(dev)
        opt=torch.optim.Adam(model.parameters(),lr=.001);gen=torch.Generator(device=dev).manual_seed(116)
        for step in range(600):
            ids=torch.randint(len(features['train']),(512,),device=dev,generator=gen)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(features['train'][ids]),arrays['train']['target'][ids])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            y=arrays['valid']['target'];pred=model(features['valid']).sigmoid();brier=(pred-y).square().mean(-1)
            rate=arrays['train']['target'].mean(0)
            row=dict(variant=variant,validation_brier=float(brier.mean()),validation_bce=float(torch.nn.functional.binary_cross_entropy(pred,y)),
                train_prevalence_baseline_brier=float((rate-y).square().mean()),per_sensor_brier=(pred-y).square().mean(0).tolist(),
                episode_brier={str(int(e)):float(brier[arrays['valid']['episode']==e].mean()) for e in arrays['valid']['episode'].unique()},
                parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=600)
            results.append(row);write_json(out/'progress.json',results);print(json.dumps(row),flush=True)
    write_json(out/'result.json',dict(results=results,episodes=episodes,rows={s:len(x['h']) for s,x in arrays.items()},
        horizon_steps=4,history_frames=10,projected_context_dim=64,contact_slice=[44,49],state_model_sha256=state_hash,
        behavior_policy=behavior,source=str(root),source_sha256=file_digest(root/'episodes.h5'),seconds=time.time()-start,
        limits=['10Hz trajectories; behavior_policy/source identify the controller; not random-policy 50Hz pen data',
          'future noisy tactile sensor bits, not ground-truth contacts or policy recovery',
          'encoder context is a modality control, not equal information history',
          'same MLP capacity, but history-only extra inputs are zero; fixed random projection is lossy',
          'single probe seed and small episode holdout; no policy benefit or causal attribution established']))
if __name__=='__main__':main()
