"""Geometric contact refinement; physical acceptance remains mandatory."""
import argparse,json,time,shutil
from pathlib import Path
import torch
from unionmimic.hand_kinematics import HandKinematics
from unionmimic.motions import qrot
from unionmimic.common import write_json,file_digest
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--input',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--distance',type=float,default=.012);a=ap.parse_args()
    root=Path(a.input).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    start=time.time();dev='cuda:%d'%a.gpu;manifest=json.loads((root/'manifest.json').read_text())
    data=[torch.load(root/c['file'],map_location=dev) for c in manifest['clips']]
    fk=HandKinematics(ROOT/'work/upstreams/WM-Craftnet/assets/urdf/right_sharpa_wave/right_sharpa_wave.urdf',manifest['joint_names'],dev)
    constants=json.loads((ROOT/'work/runs/sharpa-leap-12h-20261103-v1/task-constants.json').read_text())
    low=torch.tensor(constants['lower'],device=dev);high=torch.tensor(constants['upper'],device=dev);mid=(low+high)/2;half=(high-low)/2
    q0=torch.cat([c['q'] for c in data]);pose=torch.cat([c['pose'] for c in data]);axis=qrot(pose[:,3:],torch.tensor([1.,0,0],device=dev).expand(len(pose),-1))
    names=['right_'+f+'_fingertip' for f in ('thumb','index','middle','ring','pinky')]
    p0=fk.forward(q0,names).detach();delta=p0-pose[:,None,:3];along=(delta*axis[:,None]).sum(-1).clamp(-.105,.105)
    radial=delta-along[:,:,None]*axis[:,None];dist=radial.norm(dim=-1)
    # Thumb and two closest other fingers: preserve contact-side geometry rather than
    # dragging every finger into the cylinder. This is a candidate, not a label.
    selected=dist[:,1:].topk(2,largest=False).indices+1
    mask=torch.zeros_like(dist);mask[:,0]=1;mask.scatter_(1,selected,1)
    target=pose[:,None,:3]+along[:,:,None]*axis[:,None]+radial/dist[:,:,None].clamp_min(1e-5)*a.distance
    raw=torch.nn.Parameter(torch.atanh(((q0-mid)/half).clamp(-.999,.999)));opt=torch.optim.Adam([raw],lr=.025)
    for i in range(300):
        q=mid+half*raw.tanh();tip=fk.forward(q,names)
        contact=((tip-target)*100).square().sum(-1)
        loss=(contact*mask).sum()/mask.sum()+.05*(q-q0).square().mean()
        opt.zero_grad();loss.backward();assert torch.isfinite(raw.grad).all();opt.step()
    q=(mid+half*raw.tanh()).detach();offset=0
    for c,info in zip(data,manifest['clips']):
        n=len(c['q']);c['q']=q[offset:offset+n].cpu();offset+=n
        c={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in c.items()}
        c['contact_refinement']='geometric candidate; not physics accepted';torch.save(c,out/info['file'])
    manifest.update(physics_verified=False,contact_refinement=dict(distance_m=a.distance,loss=float(loss),seconds=time.time()-start,
        parent=str(root),parent_manifest_sha256=file_digest(root/'manifest.json'),source_sha256=file_digest(Path(__file__).resolve())))
    write_json(out/'manifest.json',manifest);print(json.dumps(manifest['contact_refinement']),flush=True)
if __name__=='__main__':main()
