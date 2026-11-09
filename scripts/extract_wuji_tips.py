"""Recover fingertip positions from the pinned Wuji robot, checking stored links."""
import argparse,json,ast
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    upstream=ROOT/'work/upstreams/dex-supermimic'
    source=ast.parse((upstream/'src/dex_supermimic/assets/wuji_hand2.py').read_text())
    mapping=next(ast.literal_eval(x.value) for x in source.body if isinstance(x,ast.AnnAssign) and getattr(x.target,'id',None)=='WUJI_HAND2_TASK_NAMES')
    model=mujoco.MjModel.from_xml_path(str(upstream/'packages/wuji-description/src/wuji_description/resources/hand2/hand2_2/body/mjcf/right.xml'))
    data=mujoco.MjData(model)
    joints=list(mapping['joint']);assert len(joints)==20
    addr=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in joints]
    bodies=[x for x,t in mapping['body'].items() if t=='right_palm_link' or ('_link' in t and 'finger' in t)]
    bids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,x) for x in bodies]
    tips=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'r_'+f+'_tip') for f in ('thumb','index_finger','middle_finger','ring_finger','pinky')]
    assert min(tips)>=0 and len(bids)==21
    wrist=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'r_wrist');rows=[]
    for group in ('02_continuous_single_axis','03_continuous_multi_axis'):
        root=ROOT/'work/reference-audits/pen-motion-20261103/clips'/group
        index=json.loads((root/'index.json').read_text())
        for entry in index['clips']:
            d=np.load(root/entry['file']);q=d['reference_qpos'];tiprows=[];errs=[]
            for i,angles in enumerate(q):
                data.qpos[addr]=angles;mujoco.mj_forward(model,data)
                R=data.xmat[wrist].reshape(3,3);p=data.xpos[wrist]
                tiprows.append((data.site_xpos[tips]-p)@R)
                w=d['wrist_pose_w'][i];rw=Rotation.from_quat(w[[4,5,6,3]]).as_matrix()
                stored=(d['link_pos_w'][i]-w[:3])@rw
                errs.append(np.max(np.linalg.norm((data.xpos[bids]-p)@R-stored,axis=-1)))
            worst=float(max(errs));assert worst<1e-4,(entry['clip_id'],worst)
            name=group+'__'+entry['clip_id']+'.npz';np.savez_compressed(out/name,tips=np.asarray(tiprows,dtype=np.float32))
            rows.append(dict(file=name,frames=len(q),stored_link_max_error_m=worst))
    (out/'manifest.json').write_text(json.dumps(dict(clips=rows,reference_commit='2becb7380fed86171056f9bf4338b2a6a0c14b01'),indent=2))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
