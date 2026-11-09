"""Exploratory imminent-failure readout on new stress episodes, no policy updates."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from unionmimic.common import write_json,file_digest

def ranking(pred, target):
    y=target.cpu().numpy();s=pred.cpu().numpy();n=len(y);pos=int(y.sum())
    if pos==0 or pos==n:return dict(positive=pos,total=n,average_precision=None,auc=None)
    # Threshold grouped rankings handle exact score ties rather than arbitrary row order.
    order=np.argsort(-s,kind='stable');ys=y[order];scores=s[order];ends=np.r_[np.where(np.diff(scores)!=0)[0],n-1]
    tp=np.cumsum(ys)[ends];fp=ends+1-tp;recall=tp/pos;precision=tp/(ends+1)
    ap=float(np.sum(np.diff(np.r_[0,recall])*precision))
    auc=float(np.trapz(np.r_[0,tp/pos],np.r_[0,fp/(n-pos)]))
    top=max(1,int(np.ceil(.05*n)))
    return dict(positive=pos,total=n,prevalence=pos/n,average_precision=ap,auc=auc,precision_top5=float(ys[:top].mean()),recall_top5=float(ys[:top].sum()/pos))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=3)
    a=p.parse_args();root=Path(a.dataset).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);dev=torch.device('cuda',a.gpu);rows=[];episode_meta=[];started=time.time()
    with h5py.File(root/'episodes.h5') as f:
        assert f.attrs['complete']
        for key,g in f.items():
            prop=g['prop'][:];n=len(prop);ix=np.arange(9,n-1)
            if not len(ix):continue
            # Failure before a window ends is observed; timeout-censored negative windows are excluded.
            failure=g['failure'][:].astype(float);fall=g['fall'][:].astype(float)
            # Include t+10, with future samples never passed as features.
            target=np.stack([np.stack([v[np.minimum(ix+j,n-1)] for j in range(1,11)],1).max(1) for v in (failure,fall)],1)
            known=(ix+10<n)|(target[:,0]>0);ix=ix[known];target=target[known]
            history=np.stack([prop[ix-j] for j in reversed(range(10))],1).reshape(len(ix),730)
            rows.append(dict(history=history,context=g['post_deter'][:][ix],target=target,episode=np.full(len(ix),int(key)),outer=np.full(len(ix),g.attrs['split']=='valid')))
            episode_meta.append((int(key),g.attrs['split'],bool(failure[-1])))
    data={k:torch.tensor(np.concatenate([r[k] for r in rows]),device=dev) for k in rows[0]}
    outer=data['outer'].bool();train=~outer;inner_ids=[]
    for failed in (False,True):
        ids=sorted(e for e,s,f in episode_meta if s=='train' and f==failed)
        if len(ids)>=2:inner_ids+=ids[1::4]
    inner=torch.isin(data['episode'],torch.tensor(inner_ids,device=dev))&train;fit=train&~inner
    assert inner.any() and fit.any() and outer.any()
    y=data['target'].double();gen=torch.Generator(device=dev).manual_seed(71);shuffled=data['context'].clone()
    for mask in (fit,inner,outer):
        x=shuffled[mask];shuffled[mask]=x[torch.randperm(len(x),device=dev,generator=gen)]
    features=dict(history=data['history'],history_wsm=torch.cat((data['history'],data['context']),1),history_shuffled=torch.cat((data['history'],shuffled),1))
    def predict(x,tr,te,alpha):
        mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(.01)
        a=((x[tr]-mu)/sd).clamp(-10,10);b=((x[te]-mu)/sd).clamp(-10,10);ym=y[tr].mean(0)
        w=torch.linalg.solve(a.T@a/len(a)+alpha*torch.eye(a.shape[1],device=dev,dtype=x.dtype),a.T@(y[tr]-ym)/len(a))
        return b@w+ym
    results=[]
    for name,x in features.items():
        x=x.double();alphas=[.001,.01,.1,1.,10.,100.];scores=[]
        for alpha in alphas:
            p=predict(x,fit,inner,alpha);scores.append([ranking(p[:,j],y[inner,j])['average_precision'] for j in range(2)])
        metrics=[]
        for j,target in enumerate(['failure_within_10','fall_within_10']):
            usable=[(v[j],i) for i,v in enumerate(scores) if v[j] is not None]
            if not usable:
                metrics.append(dict(target=target,verified=False,reason='No positive/negative inner-label support'));continue
            _,best=max(usable);pred=predict(x,train,outer,alphas[best])[:,j];rank=ranking(pred,y[outer,j])
            metrics.append(dict(target=target,verified=rank['average_precision'] is not None,alpha=alphas[best],
                brier=float((pred.clamp(0,1)-y[outer,j]).square().mean()),**rank))
        results.append(dict(variant=name,metrics=metrics));write_json(out/'progress.json',results)
    write_json(out/'result.json',dict(results=results,source=str(root),source_sha256=file_digest(root/'episodes.h5'),seconds=time.time()-started,
        split={n:sorted(set(data['episode'][m].cpu().tolist())) for n,m in [('fit',fit),('inner',inner),('outer',outer)]},
        positive={n:y[m].sum(0).cpu().tolist() for n,m in [('fit',fit),('inner',inner),('outer',outer)]},
        protocol='Linear ridge, regularization picked by inner episode AP, untouched outer episodes. Causal input only. Full ten-step windows or an earlier observed terminal failure; timeout-censored negatives excluded.',
        limits='Exploratory stress-data diagnostic; correlated time rows, few episodes and one behavior seed; information not causal control benefit. Failure includes stall, fall is reported separately.'))
if __name__=='__main__':main()
