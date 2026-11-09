"""Frozen/random/modality controls, exportable position head, disjoint bridge fit."""
from pathlib import Path
import unionmimic.runtime
import argparse,json,copy,time
import torch,numpy as np,h5py
from isaacgymenvs.utils.world_model import WorldModelAdapter
from unionmimic.models import load_world_model,state_model_digest
from unionmimic.common import UPSTREAM,RELEASE,write_json,file_digest,digest,save_checkpoint
from unionmimic.geometry_context import PositionReadout,PositionErrorBridge


def load_rows(root, stride):
    rows=[]
    with h5py.File(root/'episodes.h5') as f:
        assert f.attrs['complete'] and f.attrs['diagnostic_state_labels_only']
        for key,g in f.items():
            ix=np.arange(9,len(g['prop']),stride)
            if not len(ix):continue
            prop=g['prop'][:]
            rows.append(dict(history=np.stack([prop[ix-j] for j in reversed(range(10))],1).reshape(len(ix),730),
                embed=g['embed'][:][ix],prop=prop[ix],depth=g['depth'][:][ix],target=g['object_position_local'][:][ix],
                episode=np.full(len(ix),int(key)),step=ix,outer=np.full(len(ix),g.attrs['split']=='valid')))
    return {k:torch.as_tensor(np.concatenate([r[k] for r in rows])) for k in rows[0]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--calibration',required=True)
    p.add_argument('--test',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=1)
    a=p.parse_args();roots={k:Path(getattr(a,k)).resolve() for k in ('train','calibration','test')}
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.time()
    torch.set_num_threads(4);torch.cuda.set_device(a.gpu);dev=torch.device('cuda',a.gpu)
    datasets={k:load_rows(root,1 if k=='calibration' else 4) for k,root in roots.items()}
    wm,ck,cfg=load_world_model(roots['train']/'config.yaml',gpu=a.gpu);del ck;wm.requires_grad_(False)
    body_hash=digest(wm.encoder)
    for root in roots.values():
        with h5py.File(root/'episodes.h5') as f:assert f.attrs['state_model_sha256']==state_model_digest(wm)
    fit_ids=datasets['train']['episode'][~datasets['train']['outer']].unique().sort().values
    inner=torch.isin(datasets['train']['episode'],fit_ids[1::4])&~datasets['train']['outer']
    train_mask=~datasets['train']['outer'];fit_mask=train_mask&~inner
    def encode(encoder, rows, no_depth=False):
        result=[]
        with torch.no_grad():
            for i in range(0,len(rows['prop']),128):
                image=rows['depth'][i:i+128].to(dev)
                if no_depth:image=torch.full_like(image,.5)
                result.append(encoder(dict(prop=rows['prop'][i:i+128].to(dev),image=image)).cpu())
        return torch.cat(result)
    # Verify the stored feature contract on actual frames before any controls.
    tiny={k:v[:8] for k,v in datasets['train'].items()}
    reproduction=float((encode(wm.encoder,tiny)-tiny['embed']).abs().max());assert reproduction<2e-4
    variants=['history','pretrained','no_depth','random_501','random_502','random_503'];results=[];all_predictions={}
    y=datasets['train']['target'].to(dev).double();train_mask=train_mask.to(dev);fit_mask=fit_mask.to(dev);inner=inner.to(dev)
    alphas=[.0001,.001,.01,.1,1.,10.,100.]
    for variant in variants:
        encoder=None
        if variant.startswith('random'):
            torch.manual_seed(int(variant.split('_')[1]));wcfg=copy.deepcopy(cfg['task']['env']['cameraPolicy']['worldModel'])
            wcfg.update(replay_capacity=2,train_steps_per_epoch=0)
            adapter=WorldModelAdapter(UPSTREAM,1,22,73,(72,96,1),wcfg,dev)
            encoder=adapter.world_model.encoder.eval().requires_grad_(False);del adapter
        features={}
        for split,rows in datasets.items():
            if variant=='history':x=rows['history']
            else:
                embed=rows['embed'] if variant=='pretrained' else encode(encoder or wm.encoder,rows,variant=='no_depth')
                x=torch.cat((rows['history'],embed),-1)
            features[split]=x.to(dev).double()
        x=features['train']
        def fit(mask,alpha):
            xx=x[mask];mu=xx.mean(0);sd=xx.std(0).clamp_min(.01);xx=((xx-mu)/sd).clamp(-10,10);ym=y[mask].mean(0)
            if xx.shape[1]>len(xx):
                dual=torch.linalg.solve(xx@xx.T/len(xx)+alpha*torch.eye(len(xx),device=dev,dtype=xx.dtype),y[mask]-ym)
                w=xx.T@dual/len(xx)
            else:
                w=torch.linalg.solve(xx.T@xx/len(xx)+alpha*torch.eye(xx.shape[1],device=dev,dtype=xx.dtype),xx.T@(y[mask]-ym)/len(xx))
            return mu,sd,w,ym
        candidates=[]
        for alpha in alphas:
            mu,sd,w,ym=fit(fit_mask,alpha);pred=((x[inner]-mu)/sd).clamp(-10,10)@w+ym
            candidates.append(float((pred-y[inner]).square().mean()))
        alpha=alphas[int(np.argmin(candidates))];mu,sd,w,ym=fit(train_mask,alpha)
        head=PositionReadout(mu,sd,w,ym,variant!='history').to(dev)
        export_error=float((head(features['train'][:64,:730].float(),features['train'][:64,730:].float())-(((x[:64]-mu)/sd).clamp(-10,10)@w+ym)).abs().max())
        assert export_error<1e-6
        meta=dict(variant=variant,encoder_sha256=digest(encoder) if encoder is not None else body_hash,
                  release_sha256=file_digest(RELEASE),label='current object center in hand-base frame, metres',history_order='10 frames oldest to newest',alpha=alpha)
        save_checkpoint(out/(variant+'.pt'),head.checkpoint(**meta))
        if encoder is not None:save_checkpoint(out/(variant+'-encoder.pt'),dict(state_dict=encoder.state_dict()))
        metrics={};predictions={}
        for split,xs in features.items():
            pred=head(xs[:,:730].float(),xs[:,730:].float()).detach().cpu();predictions[split]=pred
            error=(pred-datasets[split]['target']).norm(dim=-1)*1000
            valid=datasets[split]['outer'] if split=='train' else torch.ones(len(error),dtype=torch.bool)
            metrics[split]=dict(rows=int(valid.sum()),mean_mm=float(error[valid].mean()),p95_mm=float(torch.quantile(error[valid],.95)),
                episodes={str(int(e)):float(error[valid&(datasets[split]['episode']==e)].mean()) for e in datasets[split]['episode'][valid].unique()})
        if variant=='pretrained':all_predictions=predictions
        result=dict(variant=variant,alpha=alpha,export_max_error_m=export_error,readout_coefficients=w.numel()+ym.numel(),metrics=metrics)
        results.append(result);write_json(out/'progress.json',results);print(json.dumps(result),flush=True)
        del features,head,encoder
    # A separate seed calibrates the noise process; held-out test cannot tune it.
    cal=datasets['calibration'];error=all_predictions['calibration']-cal['target'];mean=error.mean(0)
    pair=(cal['episode'][1:]==cal['episode'][:-1])&(cal['step'][1:]==cal['step'][:-1]+1)
    previous=error[:-1][pair]-mean;next_=error[1:][pair]-mean
    rho=((previous*next_).sum(0)/previous.square().sum(0).clamp_min(1e-12)).clamp(-.99,.99)
    innovation=next_-previous*rho;cov=innovation.T@innovation/len(innovation)+torch.eye(3)*1e-10
    bridge=PositionErrorBridge(mean,rho,torch.linalg.cholesky(cov))
    test_error=all_predictions['test']-datasets['test']['target']
    radius=torch.quantile(error.norm(dim=-1),.95);coverage=float((test_error.norm(dim=-1)<=radius).float().mean())
    # Derivative wrt the physical prediction must remain exactly identity.
    position=torch.zeros(16,3,requires_grad=True);observed,_=bridge.step(position,torch.zeros_like(position),torch.zeros_like(position));observed.sum().backward()
    assert torch.equal(position.grad,torch.ones_like(position))
    save_checkpoint(out/'bridge.pt',bridge.checkpoint(calibration_source=str(roots['calibration']),control_dt=.10002,nominal95_radius_m=float(radius)))
    write_json(out/'bridge.json',dict(kind='AR1 residual candidate, not calibrated confidence',rho=rho.tolist(),bias_mm=(mean*1000).tolist(),
        innovation_std_mm=(cov.diag().sqrt()*1000).tolist(),calibration95_radius_mm=float(radius*1000),heldout_radius_coverage=coverage,
        input_gradient_identity=True,limits='Marginal and temporal error approximation only; multi-step physical F prediction and closed-loop control not yet validated.'))
    # Paired per-episode differences on the external test seed, descriptive only.
    for result in results:
        result['test_delta_vs_history_mm']=result['metrics']['test']['mean_mm']-results[0]['metrics']['test']['mean_mm']
    write_json(out/'result.json',dict(results=results,seconds=time.time()-started,encoder_reproduction_error=reproduction,
        sources={k:dict(path=str(v),sha256=file_digest(v/'episodes.h5')) for k,v in roots.items()},
        trainable='position readout only; frozen encoder',head_kind='full-feature ridge, no random projection',
        controls='Same depth/prop and identical readout size for pretrained vs three random frozen encoders; depth-clamped .5 ablation.',
        limits='One training behavior policy; independent calibration/test simulation seeds, no policy-control benefit claim.'))

if __name__=='__main__':main()
