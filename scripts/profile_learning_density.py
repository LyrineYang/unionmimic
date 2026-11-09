#!/usr/bin/env python3
"""Bounded learner-only CUDA timeline diagnostic; never writes trained weights."""
import argparse,copy,gc,json,time
from pathlib import Path
import torch
from torch.profiler import profile,ProfilerActivity
from unionmimic.common import ROOT,write_json
from unionmimic.online_baseline.core import Actor,Critic,Replay
from unionmimic.online_baseline.task import HandTask,PhysicalModel,DS,DC
from unionmimic.online_fast.learner import Learner


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output).resolve();start=time.time();torch.set_num_threads(1)
    torch.cuda.set_device(a.gpu);torch.cuda.set_per_process_memory_fraction(.35,a.gpu);dev='cuda:%d'%a.gpu
    ck=torch.load(ROOT/'work/runs/sm-candidate-12h-20261102-v1/sm-detach-warm/checkpoints/segment-07.pt',map_location='cpu')
    report=dict(status='running',gpu=a.gpu,started_unix=start,horizon=16,rows=[],
      limitation='Learner-only profiling with overhead; CUDA event interval union is this process activity, not whole-GPU utilization or end-to-end learning speed.')
    write_json(out,report)
    for batch in [256,2048,8192]:
        task=HandTask(ck['task_constants']).to(dev);cfg=dict(ck['config'],batch=batch,horizon=16,failure_fraction=0.,critic_norm_source='mixed')
        obj=Learner(Actor().to(dev),Critic(DS+DC).to(dev),PhysicalModel(task).to(dev),task,cfg)
        obj.load_state_dict(copy.deepcopy(ck['learner']));obj.deadline=start+110
        replay=Replay(ck['replay']['capacity'],ck['replay']['nenv'],dev);replay.load_state_dict(ck['replay'])
        obj.fit(replay,1);obj.imagine(replay);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:
            obj.fit(replay);obj.imagine(replay);torch.cuda.synchronize()
        wall=time.perf_counter()-begin
        intervals=sorted((e.time_range.start,e.time_range.end) for e in prof.events() if str(e.device_type)=='DeviceType.CUDA')
        merged=[]
        for l,h in intervals:
            if not merged or l>merged[-1][1]:merged.append([l,h])
            else:merged[-1][1]=max(merged[-1][1],h)
        active=sum(h-l for l,h in merged)/1e6
        avg=prof.key_averages();top=sorted(avg,key=lambda e:e.self_cpu_time_total,reverse=True)[:12]
        rows=[dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000) for e in top]
        item=sum(e.count for e in avg if e.key=='aten::_local_scalar_dense')
        row=dict(batch=batch,profile_wall_seconds=wall,cuda_device_active_seconds=active,
            cuda_activity_fraction_with_profiler=active/wall,cuda_event_count=len(intervals),scalar_host_reads=item,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,cpu_top=rows)
        report['rows'].append(row);write_json(out,report)
        del prof,obj,replay,task;gc.collect();torch.cuda.empty_cache()
    report.update(status='completed',ended_unix=time.time());write_json(out,report)

if __name__=='__main__':main()
