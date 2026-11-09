"""Assemble all four calibration lanes without discarding any reference clip."""
import argparse,json,hashlib
from pathlib import Path
import torch,numpy as np
from unionmimic.common import write_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--calibration',required=True);p.add_argument('--out',required=True)
    p.add_argument('--prefix');p.add_argument('--lanes',type=int,default=4)
    a=p.parse_args();root=Path(a.input).resolve();cal=Path(a.calibration).resolve();out=Path(a.out).resolve()
    manifest=json.loads((root/'manifest.json').read_text());source=file_digest(root/'manifest.json');rows=[]
    anchors=None
    for gpu in range(a.lanes):
        info=json.loads((cal/f'gpu{gpu}/result.json').read_text());assert info['completed'] and info['source_manifest_sha256']==source
        count=info.get('anchors_per_clip',32)
        if anchors is None:anchors=count
        assert anchors==count
        rows.append(torch.load(cal/f'gpu{gpu}/anchors.pt',map_location='cpu'))
    data={k:torch.cat([r[k] for r in rows]) for k in rows[0]};order=data['nodes'].argsort();data={k:v[order] for k,v in data.items()}
    assert torch.equal(data['nodes'],torch.arange(13*anchors)),'Missing/duplicate anchors'
    out.mkdir(parents=True,exist_ok=False)
    import xml.etree.ElementTree as ET
    repo=Path(__file__).resolve().parents[1];limits=json.loads((repo/'work/runs/sharpa-leap-12h-20261103-v1/task-constants.json').read_text())
    low=torch.tensor(limits['lower']);high=torch.tensor(limits['upper'])
    for i,info in enumerate(manifest['clips']):
        d=torch.load(root/info['file'],map_location='cpu');mask=data['clips']==i;x=data['phases'][mask].numpy();timeline=np.arange(len(d['q']))
        def interpolate(key):
            y=data[key][mask].numpy();return torch.tensor(np.stack([np.interp(timeline,x,y[:,j]) for j in range(y.shape[1])],1),dtype=torch.float32)
        qoff=interpolate('q_offset');poff=interpolate('position_offset');assert poff.abs().max()<=.005001
        d['q']=(d['q']+qoff).clamp(low,high);d['pose'][:,:3]+=poff;d['sharpa_physics_verified']=False
        if a.prefix:
            prefix=Path(a.prefix).resolve();pm=json.loads((prefix/'manifest.json').read_text())
            assert pm['clips'][i]['id']==info['id']
            pd=torch.load(prefix/pm['clips'][i]['file'],map_location='cpu')
            assert torch.equal(pd['pose'][:,3:],d['pose'][:,3:]) and pd['frame_dt']==d['frame_dt']
            # Keep the entire independently optimized 1.5-second initial segment;
            # blend corrections over the next second without changing timestamps.
            weight=((125-torch.arange(len(d['q']))).float()/50).clamp(0,1)[:,None]
            d['q']=weight*pd['q']+(1-weight)*d['q']
            d['pose'][:,:3]=weight*pd['pose'][:,:3]+(1-weight)*d['pose'][:,:3]
        torch.save(d,out/info['file']);info['retarget_sha256']=file_digest(out/info['file'])
    manifest.update(physics_verified=False,phase_calibration=dict(parent_manifest_sha256=source,anchors=13*anchors,
        lane_results=[str(cal/f'gpu{i}/result.json') for i in range(a.lanes)],q_interpolation='piecewise linear correction, hardware clipped',
        position_correction_bound_per_axis_m=.005,object_rotations_and_timestamps_unchanged=True,independent_midpoint_acceptance_pending=True,
        initial_segment_parent=a.prefix,initial_segment_manifest_sha256=file_digest(Path(a.prefix)/'manifest.json') if a.prefix else None,
        initial_segment_preserved_seconds=1.5 if a.prefix else None,initial_segment_blend_end_seconds=2.5 if a.prefix else None))
    write_json(out/'manifest.json',manifest);print(out)
if __name__=='__main__':main()
