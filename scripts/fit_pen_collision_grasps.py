"""Collision-aware candidate reset grasps, followed by independent physics checks.

Fits against the Sharpa URDF collision meshes and a finite X-axis cylinder.
This is offline initialization preparation, never an action teacher for SM/PPO.
"""
import argparse,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch,trimesh
from unionmimic.hand_kinematics import HandKinematics,rpy_quat
from unionmimic.motions import qrot,qinv
from unionmimic.common import write_json,file_digest
ROOT=Path(__file__).resolve().parents[1]

def cylinder_sdf(points):
    radial=points[...,1:].norm(dim=-1)-.0075;axial=points[...,0].abs()-.11
    d=torch.stack((radial,axial),-1)
    return d.clamp_min(0).norm(dim=-1)+d.max(-1).values.clamp_max(0)

def mesh_points(urdf,device):
    xml=ET.parse(urdf).getroot();meshes={}
    for link in xml.findall('link'):
        collision=link.find('collision')
        if collision is None:continue
        mesh=collision.find('geometry/mesh')
        if mesh is None:continue
        path=urdf.parent/'meshes'/Path(mesh.get('filename')).name
        m=trimesh.load(path,force='mesh',process=False)
        points=np.unique(m.vertices,axis=0)
        # Deterministic coverage, independent of the physics validation seed.
        points=points[np.linspace(0,len(points)-1,min(128,len(points))).astype(int)]
        p=torch.tensor(points,device=device,dtype=torch.float32)
        origin=collision.find('origin')
        if origin is not None:
            xyz=torch.tensor([float(v) for v in origin.get('xyz','0 0 0').split()],device=device)
            rot=torch.tensor(rpy_quat(*[float(v) for v in origin.get('rpy','0 0 0').split()]),device=device)
            p=qrot(rot.expand(len(p),-1),p)+xyz
        meshes[link.get('name')]=p
    assert all('right_'+f+'_elastomer' in meshes for f in ('thumb','index','middle','ring','pinky'))
    return meshes

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--hardware-limits',action='store_true');ap.add_argument('--contact-margin',type=float,default=.001)
    ap.add_argument('--iterations',type=int,default=600);ap.add_argument('--starts',type=int,default=8);a=ap.parse_args()
    root=Path(a.input).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    dev='cuda:%d'%a.gpu;torch.manual_seed(20261111);start=time.time();manifest=json.loads((root/'manifest.json').read_text())
    urdf=ROOT/'work/upstreams/WM-Craftnet/assets/urdf/right_sharpa_wave/right_sharpa_wave.urdf'
    source=[torch.load(root/c['file'],map_location=dev) for c in manifest['clips']]
    q0=torch.stack([x['q'][0] for x in source]);pose=torch.stack([x['pose'][0] for x in source])
    fk=HandKinematics(urdf,manifest['joint_names'],dev);meshes=mesh_points(urdf,dev)
    constants=json.loads((ROOT/'work/runs/sharpa-leap-12h-20261103-v1/task-constants.json').read_text())
    low=torch.tensor(constants['lower'],device=dev);high=torch.tensor(constants['upper'],device=dev)
    if a.hardware_limits:
        limits={j.get('name'):j.find('limit') for j in ET.parse(urdf).findall('joint') if j.get('type')=='revolute'}
        low=torch.tensor([float(limits[n].get('lower'))*.9 for n in manifest['joint_names']],device=dev)
        high=torch.tensor([float(limits[n].get('upper'))*.9 for n in manifest['joint_names']],device=dev)
    mid=(low+high)/2;half=(high-low)/2;n=13*a.starts
    initial=q0.repeat_interleave(a.starts,0);pose=pose.repeat_interleave(a.starts,0)
    guess=initial+torch.randn_like(initial)*.08;guess[::a.starts]=q0
    raw=torch.nn.Parameter(torch.atanh(((guess-mid)/half).clamp(-.999,.999)));opt=torch.optim.Adam([raw],lr=.015)
    pads=['right_'+f+'_elastomer' for f in ('thumb','index','middle','ring','pinky')]
    # Choose the two closest non-thumb pads once. Do not switch correspondences
    # during optimization or drag all fingers through the pen.
    selected=None
    for it in range(a.iterations):
        q=mid+half*raw.tanh();positions,rotations=fk.forward(q);inside=[];pad_dist=[];pad_centers=[]
        for name,local in meshes.items():
            world=qrot(rotations[name][:,None].expand(-1,len(local),-1),local[None].expand(n,-1,-1))+positions[name][:,None]
            pen=qrot(qinv(pose[:,3:])[:,None].expand(-1,len(local),-1),world-pose[:,None,:3])
            sdf=cylinder_sdf(pen)
            inside.append((-sdf-.0003).clamp_min(0).square().mean(-1))
            if name in pads:
                pad_dist.append((pads.index(name),(sdf-a.contact_margin).abs().min(-1).values))
                pad_centers.append((pads.index(name),pen.mean(1)[:,1:]))
        d=torch.stack([v for _,v in sorted(pad_dist)],1);centers=torch.stack([v for _,v in sorted(pad_centers)],1)
        if selected is None:selected=d[:,1:].detach().topk(2,largest=False).indices+1
        chosen=torch.cat((torch.zeros(n,1,dtype=torch.long,device=dev),selected),1)
        contact=d.gather(1,chosen).square().mean(-1)
        centers=torch.nn.functional.normalize(centers,dim=-1)
        thumb=centers[:,0];others=centers.gather(1,selected[:,:,None].expand(-1,-1,2)).mean(1)
        opposed=((thumb*others).sum(-1)+.2).clamp_min(0).square()
        loss_rows=1000000*torch.stack(inside,1).sum(1)+20000*contact+.1*opposed+.05*(q-initial).square().mean(-1)
        loss=loss_rows.mean();opt.zero_grad();loss.backward();assert torch.isfinite(raw.grad).all();opt.step()
        if it%100==0:print(json.dumps(dict(iteration=it,loss=float(loss),seconds=time.time()-start)),flush=True)
    q=(mid+half*raw.tanh()).detach();choice=loss_rows.detach().view(13,a.starts).argmin(1)+torch.arange(13,device=dev)*a.starts
    best=q[choice].cpu()
    for i,(c,info) in enumerate(zip(source,manifest['clips'])):
        c={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in c.items()}
        delta=best[i]-c['q'][0];k=min(50,len(c['q']));blend=(1-torch.arange(k).float()/(k-1))[:,None]
        c['q'][:k]+=delta*blend;c['q']=c['q'].clamp(low.cpu(),high.cpu());c['sharpa_physics_verified']=False
        # Parent keypoint positions do not describe the newly fitted joints.
        c['parent_actual_links']=c.pop('actual_links',None)
        torch.save(c,out/info['file'])
    manifest.update(physics_verified=False,geometry_only=True,pen_limits='hardware_90pct' if a.hardware_limits else 'original_block_limits',
        collision_fit=dict(cli=vars(a),seconds=time.time()-start,source_sha256=file_digest(Path(__file__).resolve()),
            mesh_points=sum(len(x) for x in meshes.values()),selected_objectives=loss_rows[choice].detach().cpu().tolist(),
            parent_manifest_sha256=file_digest(root/'manifest.json'),method='sampled collision surface, not an exact signed mesh collision check'))
    write_json(out/'manifest.json',manifest);print(json.dumps(dict(done=True,seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
