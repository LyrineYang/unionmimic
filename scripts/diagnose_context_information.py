"""Nested episode-held-out readouts: projection loss, overfit and target relevance.

Diagnostic follow-up to a negative contact probe, not metric selection to erase it.
No future action/observation is supplied to a readout; future values are labels only.
"""
import argparse,json,time
from pathlib import Path
import torch,h5py,numpy as np
from unionmimic.common import write_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.dataset).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);dev=torch.device('cuda',a.gpu);torch.cuda.set_device(dev);started=time.time()
    rows=[]
    with h5py.File(root/'episodes.h5') as f:
        assert f.attrs['complete']
        for key,g in f.items():
            prop=g['prop'][:];ix=np.arange(31,len(prop)-4,2)
            if not len(ix):continue
            hist=np.stack([prop[ix-j] for j in reversed(range(32))],1)
            rows.append(dict(history=hist[:,-10:].reshape(len(ix),730),long_history=hist.reshape(len(ix),2336),
                context=g['post_deter'][:][ix],target=np.concatenate((prop[ix+4,44:49],prop[ix+4,:22]),1),
                episode=np.full(len(ix),int(key)),outer=np.full(len(ix),g.attrs['split']=='valid')))
    data={k:torch.tensor(np.concatenate([r[k] for r in rows]),device=dev) for k in rows[0]}
    outer=data['outer'].bool();train=~outer
    eps=data['episode'][train].unique().sort().values
    inner=torch.isin(data['episode'],eps[1::4])&train;fit=train&~inner
    gen=torch.Generator(device=dev).manual_seed(113)
    projection=torch.randn(512,64,device=dev,generator=gen)/512**.5
    perm_h=data['context'].clone()
    # Shuffle within each split, not across its boundary.
    for mask in (fit,inner,outer):
        vals=perm_h[mask];perm_h[mask]=vals[torch.randperm(len(vals),device=dev,generator=gen)]
    features=dict(history=data['history'],history_32=data['long_history'],
        history_wsm64=torch.cat((data['history'],data['context']@projection),1),
        history_wsm512=torch.cat((data['history'],data['context']),1),
        history_shuffled512=torch.cat((data['history'],perm_h),1))
    y=data['target'].double();results=[]
    alphas=[.0001,.001,.01,.1,1.,10.,100.]
    def fit_predict(x,training,testing,alpha):
        mu=x[training].mean(0);sd=x[training].std(0).clamp_min(.01)
        xx=((x[training]-mu)/sd).clamp(-10,10);xt=((x[testing]-mu)/sd).clamp(-10,10)
        ym=y[training].mean(0);yc=y[training]-ym
        gram=xx.T@xx/len(xx);rhs=xx.T@yc/len(xx)
        w=torch.linalg.solve(gram+alpha*torch.eye(gram.shape[0],device=dev,dtype=x.dtype),rhs)
        return xt@w+ym
    for name,x in features.items():
        x=x.double();candidates=[]
        for alpha in alphas:
            pred=fit_predict(x,fit,inner,alpha)
            candidates.append([float((pred[:,:5].clamp(0,1)-y[inner,:5]).square().mean()),
                               float((pred[:,5:]-y[inner,5:]).square().mean())])
        selected=[alphas[int(np.argmin(np.array(candidates)[:,i]))] for i in range(2)]
        contact=fit_predict(x,train,outer,selected[0])[:,:5].clamp(0,1)
        joints=fit_predict(x,train,outer,selected[1])[:,5:]
        errors=torch.stack(((contact-y[outer,:5]).square().mean(-1),(joints-y[outer,5:]).square().mean(-1)),1)
        ep_errors={str(int(e)):errors[data['episode'][outer]==e].mean(0).tolist() for e in data['episode'][outer].unique()}
        result=dict(variant=name,features=x.shape[1],selected_alpha=selected,inner_errors=candidates,
            contact_brier=float(errors[:,0].mean()),joint_rmse=float(errors[:,1].mean().sqrt()),episode_errors=ep_errors)
        results.append(result);print(json.dumps(result),flush=True);write_json(out/'progress.json',results)
    # Paired cluster bootstrap over held-out episodes, not correlated individual frames.
    base=np.array(list(results[0]['episode_errors'].values()))[:,0]
    rng=np.random.default_rng(197);indices=rng.integers(0,len(base),size=(10000,len(base)))
    for result in results:
        values=np.array(list(result['episode_errors'].values()))[:,0]-base
        result['contact_delta_bootstrap95_vs_history']=np.quantile(values[indices].mean(1),[.025,.975]).tolist()
    write_json(out/'result.json',dict(results=results,source=str(root),source_sha256=file_digest(root/'episodes.h5'),
        split={name:sorted(set(data['episode'][mask].cpu().tolist())) for name,mask in [('fit',fit),('inner',inner),('outer',outer)]},
        rows={name:int(mask.sum()) for name,mask in [('fit',fit),('inner',inner),('outer',outer)]},
        horizon=4,seconds=time.time()-started,
        limits=['Four outer episodes from one behavior policy; bootstrap is descriptive, not seed-level significance.',
                'Linear ridge capacity differs with feature count; inner episode validation selects regularization, outer untouched.',
                'Future joint prediction is a separate exploratory target. It does not replace the negative contact result.',
                'No actor/F training or student deployment here; no claim of control improvement.']))
if __name__=='__main__':main()
