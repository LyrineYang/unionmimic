#!/usr/bin/env python3
"""Bounded native collection scaling probe; no training/replay/checkpoint mutation."""
import ctypes,os,sys,traceback
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
import torch
import argparse,json,time,statistics,subprocess
from pathlib import Path
from unionmimic.common import ROOT,write_json
from unionmimic.online_baseline.core import Actor,Critic
from unionmimic.online_baseline.task import DS,DC
from unionmimic.online_baseline.simulator import make_simulator,Collector


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--envs',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).resolve();start=time.time();torch.set_num_threads(1)
    torch.cuda.set_device(a.gpu);torch.cuda.set_per_process_memory_fraction(.5,a.gpu)
    report=dict(status='running',gpu=a.gpu,envs=a.envs,started_unix=start,co_resident=True,training=False)
    write_json(out,report)
    env,task,_=make_simulator(a.envs,a.gpu,19832)
    ck=torch.load(ROOT/'work/runs/sm-candidate-12h-20261102-v1/sm-detach-warm/checkpoints/segment-07.pt',map_location=env.device)
    actor=Actor().to(env.device);critic=Critic(DS+DC).to(env.device)
    actor.load_state_dict(ck['learner']['actor']);critic.load_state_dict(ck['learner']['critic'])
    collector=Collector(env,task,actor,critic);timing=[];valid=0
    for i in range(112):
        if time.time()>start+140:raise TimeoutError('Collection probe deadline')
        torch.cuda.synchronize();tic=time.perf_counter();row=collector.step(noise=.1,normalize=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-tic
        if i>=16:timing.append(elapsed);valid+=int(row['valid'].sum())
    report.update(status='completed',ended_unix=time.time(),seconds=time.time()-start,
        measured_vector_steps=len(timing),median_step_seconds=statistics.median(timing),
        mean_step_seconds=sum(timing)/len(timing),valid_transitions_per_second=valid/sum(timing),
        torch_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        parity=collector.evidence())
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']).decode()
    report['process_gpu_memory_mib']=next((int(l.split(',')[1]) for l in raw.splitlines() if int(l.split(',')[0])==os.getpid()),None)
    write_json(out,report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    finally:
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        sys.stdout.flush();sys.stderr.flush()
    os._exit(code)
