"""Current hidden object-state readout; all simulator states are labels only."""
import argparse,time
from pathlib import Path
import numpy as np,h5py,torch
from unionmimic.common import write_json,file_digest

def rotation_error(pred,true):
    def basis(x):
        u=torch.nn.functional.normalize(x[:,:3],dim=-1)
        v=torch.nn.functional.normalize(x[:,3:]-(u*x[:,3:]).sum(-1,keepdim=True)*u,dim=-1)
        return torch.stack((u,v,torch.cross(u,v,dim=-1)),-1)
    r=basis(pred).transpose(-1,-2)@basis(true)
    return torch.acos(((r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))*180/np.pi

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=3)
    a=p.parse_args();root=Path(a.dataset).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);dev=torch.device('cuda',a.gpu);started=time.time();rows=[]
    with h5py.File(root/'episodes.h5') as f:
        assert f.attrs['complete'] and f.attrs['diagnostic_state_labels_only']
        for key,g in f.items():
            ix=np.arange(9,len(g['prop']),2)
            if not len(ix):continue
            prop=g['prop'][:];h=np.stack([prop[ix-j] for j in reversed(range(10))],1).reshape(len(ix),730)
            rows.append(dict(history=h,context=g['post_deter'][:][ix],embed=g['embed'][:][ix],
                y=np.concatenate([g[k][:][ix] for k in ['object_position_local','object_rotation6d','object_velocity_local']],-1),
                episode=np.full(len(ix),int(key)),outer=np.full(len(ix),g.attrs['split']=='valid')))
    data={k:torch.tensor(np.concatenate([r[k] for r in rows]),device=dev) for k in rows[0]}
    outer=data['outer'].bool();train=~outer;eps=data['episode'][train].unique().sort().values
    inner=torch.isin(data['episode'],eps[1::4])&train;fit=train&~inner;y=data['y'].double()
    gen=torch.Generator(device=dev).manual_seed(731)
    proj=torch.randn(data['embed'].shape[1],512,device=dev,generator=gen)/data['embed'].shape[1]**.5
    shuffle=data['context'].clone()
    for mask in (fit,inner,outer):
        v=shuffle[mask];shuffle[mask]=v[torch.randperm(len(v),device=dev,generator=gen)]
    features=dict(history=data['history'],history_encoder512=torch.cat((data['history'],data['embed']@proj),-1),
        history_wsm512=torch.cat((data['history'],data['context']),-1),history_shuffled512=torch.cat((data['history'],shuffle),-1),
        train_mean=torch.zeros(len(y),1,device=dev))
    def predict(x,tr,te,alpha):
        mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(.01);xx=((x[tr]-mu)/sd).clamp(-10,10);xt=((x[te]-mu)/sd).clamp(-10,10);ym=y[tr].mean(0)
        w=torch.linalg.solve(xx.T@xx/len(xx)+alpha*torch.eye(xx.shape[1],device=dev,dtype=x.dtype),xx.T@(y[tr]-ym)/len(xx))
        return xt@w+ym
    results=[];alphas=[.0001,.001,.01,.1,1.,10.,100.];slices=[slice(0,3),slice(3,9),slice(9,12)]
    for name,x in features.items():
        x=x.double();errors=[]
        for alpha in alphas:
            pred=predict(x,fit,inner,alpha)
            errors.append([float((pred[:,sl]-y[inner,sl]).square().mean()) for sl in slices])
        chosen=[alphas[int(np.argmin(np.array(errors)[:,j]))] for j in range(3)]
        pred=torch.cat([predict(x,train,outer,alpha)[:,sl] for alpha,sl in zip(chosen,slices)],-1)
        pos=(pred[:,:3]-y[outer,:3]).norm(dim=-1)*1000
        rot=rotation_error(pred[:,3:9],y[outer,3:9]);vel=(pred[:,9:]-y[outer,9:]).norm(dim=-1)*1000
        per_ep={str(int(e)):dict(position_mm=float(pos[data['episode'][outer]==e].mean()),rotation_deg=float(rot[data['episode'][outer]==e].mean()),velocity_mm_s=float(vel[data['episode'][outer]==e].mean())) for e in data['episode'][outer].unique()}
        results.append(dict(variant=name,selected_alpha=chosen,position_mm=float(pos.mean()),rotation_deg=float(rot.mean()),velocity_mm_s=float(vel.mean()),episode=per_ep))
    rng=np.random.default_rng(72);n=len(results[0]['episode']);ids=rng.integers(n,size=(10000,n))
    for r in results:
        for metric in ('position_mm','rotation_deg','velocity_mm_s'):
            delta=np.array([v[metric] for v in r['episode'].values()])-np.array([v[metric] for v in results[0]['episode'].values()])
            r[metric+'_delta_ci95']=np.quantile(delta[ids].mean(1),[.025,.975]).tolist()
    write_json(out/'result.json',dict(results=results,source=str(root),source_sha256=file_digest(root/'episodes.h5'),seconds=time.time()-started,
        split={n:sorted(set(data['episode'][m].cpu().tolist())) for n,m in [('fit',fit),('inner',inner),('outer',outer)]},
        rows={n:int(m.sum()) for n,m in [('fit',fit),('inner',inner),('outer',outer)]},
        target='Current hand-local hidden object state; geometry labels never fed to policy or WSM. No future inputs.',
        limits=['Exploratory new target/new dataset; does not replace previous contact result.',
                'Current encoder and recurrent WSM share pretrained source and modality; 512-d random encoder projection is lossy.',
                'Small episodic holdout, one behavior policy/seed, different effective feature capacity; not a policy benefit or independent-seed significance test.']))
if __name__=='__main__':main()
