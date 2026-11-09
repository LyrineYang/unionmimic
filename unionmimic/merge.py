import json
from pathlib import Path
import shutil
import h5py
import numpy as np
from .common import file_digest, write_json


def merge(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'episodes.h5').exists()
    folders = [Path(p).resolve() for p in args.data]
    records, sources = [], []
    expected = None
    with h5py.File(out/'episodes.h5','w') as dest:
        for folder in folders:
            with h5py.File(folder/'episodes.h5','r') as source:
                assert source.attrs['complete']
                signature = (source.attrs['source_checkpoint_sha256'],source.attrs['state_model_sha256'],source.attrs['object'])
                if expected is None:
                    expected = signature
                    for key,value in source.attrs.items():
                        dest.attrs[key]=value
                assert signature[1:] == expected[1:], 'Only identical state models/tasks can share a cache'
                for key in source:
                    new_key = '%06d' % len(records)
                    source.copy(key,dest,name=new_key)
                    g=dest[new_key]
                    g.attrs['source_episode']=str(folder)+':'+key
                    records.append(dict(id=new_key,behavior=g.attrs['behavior'],
                                        split=g.attrs.get('split',''),
                                        failure=bool(g['failure'][-1]),timeout=bool(g['timeout'][-1]),
                                        fall=bool(g['fall'][-1]),transitions=len(g['reward'])-1))
            sources.append(dict(path=str(folder),sha256=file_digest(folder/'episodes.h5')))
        rng=np.random.default_rng(args.seed)
        strata={}
        for r in records:
            if not r['split']:
                strata.setdefault((r['behavior'],r['failure']),[]).append(r)
        for group in strata.values():
            rng.shuffle(group)
            n=max(1,round(.2*len(group))) if len(group)>=3 else 0
            for i,r in enumerate(group):
                r['split']='valid' if i<n else 'train'
                dest[r['id']].attrs['split']=r['split']
        dest.attrs['sources']=json.dumps(sources)
    shutil.copyfile(folders[0]/'config.yaml',out/'config.yaml')
    report=dict(episodes=len(records),transitions=sum(r['transitions'] for r in records),
                failures=sum(r['failure'] for r in records),falls=sum(r['fall'] for r in records),
                timeouts=sum(r['timeout'] for r in records),sources=sources,
                valid_failures=sum(r['failure'] and r['split']=='valid' for r in records),records=records)
    write_json(out/'collection.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
