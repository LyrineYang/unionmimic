#!/usr/bin/env python3
"""Bounded sidecar benchmark: no simulator, no checkpoint writes or live-code edits."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path
import torch
from unionmimic.common import ROOT,write_json,file_digest
from unionmimic.online_baseline.core import Actor,Critic,Replay
from unionmimic.online_baseline.task import HandTask,PhysicalModel,DS,DC
from unionmimic.online_baseline.learner import Learner as Baseline
from unionmimic.online_fast.learner import Learner as Fast


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--horizon',type=int,default=8)
    ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output).resolve();start=time.time();torch.set_num_threads(1)
    torch.cuda.set_device(a.gpu);torch.cuda.set_per_process_memory_fraction(.25,a.gpu)
    dev='cuda:%d'%a.gpu
    p=ROOT/'work/runs/sm-candidate-12h-20261102-v1/sm-detach-warm/checkpoints/segment-07.pt'
    ck=torch.load(p,map_location='cpu')
    report=dict(started_unix=start,gpu=a.gpu,horizon=a.horizon,parent_sha256=file_digest(p),
                co_resident_with_live_training=True,allocator_cap_gib=8,settings=[],status='running')
    write_json(out,report)
    def make(kind,batch):
        task=HandTask(ck['task_constants']).to(dev)
        cfg=dict(ck['config'],batch=batch,horizon=a.horizon,failure_fraction=0.,critic_norm_source='mixed')
        obj=(Baseline if kind=='baseline' else Fast)(Actor().to(dev),Critic(DS+DC).to(dev),PhysicalModel(task).to(dev),task,cfg)
        obj.load_state_dict(copy.deepcopy(ck['learner']));obj.deadline=start+165
        r=Replay(ck['replay']['capacity'],ck['replay']['nenv'],dev);r.load_state_dict(ck['replay'])
        return obj,r
    # Exact same full fit/imagine RNG and parent for the numerical comparison.
    states=[]
    for kind in ['baseline','fast']:
        obj,r=make(kind,256);torch.manual_seed(19831);torch.cuda.manual_seed(19831)
        obj.fit(r,1);obj.imagine(r)
        states.append({n:{k:v.detach().cpu().clone() for k,v in getattr(obj,n).state_dict().items()} for n in ['actor','critic','model']})
        del obj,r;gc.collect();torch.cuda.empty_cache()
    error=max(float((states[0][n][k]-states[1][n][k]).abs().max()) for n in states[0] for k in states[0][n])
    report['full_update_max_abs_difference']=error
    if error!=0:raise AssertionError('Safety aggregation changes the actual update: '+str(error))
    for batch in [256,512,1024,2048]:
        for kind in ['baseline','fast']:
            if time.time()>start+140:break
            obj,r=make(kind,batch);torch.manual_seed(19831);torch.cuda.manual_seed(19831)
            try:
                obj.fit(r,1);obj.imagine(r);torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats();times=[]
                for _ in range(2):
                    torch.cuda.synchronize();t=time.perf_counter();obj.fit(r);torch.cuda.synchronize();m=time.perf_counter()-t
                    t=time.perf_counter();obj.imagine(r);torch.cuda.synchronize();v=time.perf_counter()-t
                    times.append(dict(model=m,actor_critic=v,total=m+v))
                row=dict(kind=kind,batch=batch,timing=times,
                    mean_seconds=sum(x['total'] for x in times)/len(times),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                row['actor_start_samples_per_second']=batch/row['mean_seconds']
                report['settings'].append(row)
            except torch.cuda.OutOfMemoryError:
                report['settings'].append(dict(kind=kind,batch=batch,oom=True))
            finally:
                del obj,r;gc.collect();torch.cuda.empty_cache()
            write_json(out,report)
    report.update(status='completed',ended_unix=time.time(),seconds=time.time()-start)
    write_json(out,report);print(json.dumps(report))

if __name__=='__main__':main()
