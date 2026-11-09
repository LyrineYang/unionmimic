"""Geometry-based 20 -> 22 DOF retargeting; reports errors, never calls it physics validation."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import torch
from unionmimic.hand_kinematics import HandKinematics
from unionmimic.motions import MotionLibrary,qmul
from unionmimic.common import write_json
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=3);ap.add_argument('--out',required=True);ap.add_argument('--iterations',type=int,default=250)
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.time()
    device='cuda:%d'%a.gpu
    source=torch.load(ROOT/'work/runs/sharpa-leap-interface-20261103-v1/transitions.pt',map_location='cpu')
    names=source['joint_names'];constants=source['task_constants'];urdf=ROOT/'work/upstreams/WM-Craftnet/assets/urdf/right_sharpa_wave/right_sharpa_wave.urdf'
    fk=HandKinematics(urdf,names,device);lib=MotionLibrary(ROOT/'work/reference-audits/pen-motion-20261103/clips',device)
    fingers=('thumb','index','middle','ring','pinky')
    target_names=['right_hand_C_MC']
    for f in fingers:
        suffix=('CMC_VL','MC','PP','DP') if f=='thumb' else ('MCP_VL','PP','MP','DP')
        target_names += ['right_'+f+'_'+s for s in suffix]
    target_rest=fk.forward(torch.zeros(1,22,device=device),target_names)[0]
    anchors=[0,1,5,9,13,17]
    src=torch.stack([c['links'][0,anchors] for c in lib.clips]).mean(0);dst=target_rest[anchors]
    sm=src.mean(0);dm=dst.mean(0);U,_,Vh=torch.linalg.svd((src-sm).T@(dst-dm));fix=torch.eye(3,device=device);fix[2,2]=torch.det(U@Vh)
    # Row vectors: source @ rotation + translation, rigid (no object scaling).
    rotation=U@fix@Vh;translation=dm-sm@rotation
    goals=torch.cat([c['links']@rotation+translation for c in lib.clips]);qsrc=torch.cat([c['source_joints'] for c in lib.clips])
    lower=torch.tensor(constants['lower'],device=device);upper=torch.tensor(constants['upper'],device=device)
    prior=torch.zeros(len(goals),22,device=device)
    mapping={'thumb':{'CMC_FE':0,'CMC_AA':1,'MCP_FE':2,'IP':3},
             **{f:{'MCP_FE':4*i,'MCP_AA':4*i+1,'PIP':4*i+2,'DIP':4*i+3} for i,f in enumerate(fingers) if i}}
    for f,d in mapping.items():
        for suffix,col in d.items():prior[:,names.index('right_'+f+'_'+suffix)]=qsrc[:,col]
    prior=prior.clamp(lower+.001,upper-.001);mid=(lower+upper)/2;half=(upper-lower)/2
    raw=torch.nn.Parameter(torch.atanh(((prior-mid)/half).clamp(-.999,.999)));optimizer=torch.optim.Adam([raw],lr=.04)
    valid_adj=torch.ones(len(goals)-1,device=device);edge=0
    for c in lib.clips[:-1]:edge+=c['frames'];valid_adj[edge-1]=0
    weights=torch.tensor([.1]+[.2,.3,1.,2.]*5,device=device)[None,:,None]
    for it in range(a.iterations):
        q=mid+half*raw.tanh();pred=fk.forward(q,target_names)
        geometric=((pred-goals)*100).square().mul(weights).mean()
        smooth=((q[1:]-q[:-1]).square().mean(-1)*valid_adj).sum()/valid_adj.sum()
        loss=geometric+.01*(q-prior).square().mean()+.2*smooth
        optimizer.zero_grad();loss.backward();assert torch.isfinite(raw.grad).all();optimizer.step()
        if it%50==0:print(json.dumps(dict(iteration=it,loss=float(loss),position_rmse_m=float((pred-goals).square().mean().sqrt()),elapsed=time.time()-start)),flush=True)
    q=(mid+half*raw.tanh()).detach();pred=fk.forward(q,target_names).detach();offset=0;clips=[]
    # Quaternion for the same rigid rotation; matrix convention uses column vectors.
    from scipy.spatial.transform import Rotation
    xyzw=Rotation.from_matrix(rotation.detach().cpu().numpy().T).as_quat();rq=torch.tensor(xyzw[[3,0,1,2]],device=device,dtype=torch.float32)
    for c in lib.clips:
        sl=slice(offset,offset+c['frames']);err=(pred[sl]-goals[sl]).norm(dim=-1)
        pose=c['pose'].clone();pose[:,:3]=pose[:,:3]@rotation+translation;pose[:,3:]=qmul(rq.expand(len(pose),-1),pose[:,3:])
        filename=c['id'].replace('/','__')+'.pt'
        torch.save(dict(q=q[sl].cpu(),pose=pose.cpu(),source_links=goals[sl].cpu(),actual_links=pred[sl].cpu(),joint_names=names,
            frame_dt=.02,source_sha256=c['sha256'],sharpa_physics_verified=False),out/filename)
        clips.append(dict(id=c['id'],file=filename,frames=c['frames'],mean_keypoint_error_m=float(err.mean()),
            p95_keypoint_error_m=float(torch.quantile(err,.95)),max_joint_speed_rad_s=float(((q[sl][1:]-q[sl][:-1])/.02).abs().max())))
        offset+=c['frames']
    write_json(out/'manifest.json',dict(clips=clips,joint_names=names,urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        rotation_row=rotation.detach().cpu().tolist(),translation=translation.detach().cpu().tolist(),
        anchor_alignment_rmse_m=float(((src@rotation+translation-dst)**2).mean().sqrt()),
        geometry_only=True,physics_verified=False,limits_satisfied=bool(((q>=lower)&(q<=upper)).all()),seconds=time.time()-start))
    print(json.dumps(dict(done=True,seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
