#!/usr/bin/env python
"""Fixed-seed evaluation: one complete episode per environment, no fast-failure selection."""
import ctypes
import os
import sys
import traceback
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from unionmimic.common import write_json,file_digest
from unionmimic.online.core import Actor,Critic
from unionmimic.online.task import DS,DC,qrotate,qconj
from unionmimic.online.simulator import make_simulator,Collector


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,default=93001)
    p.add_argument('--envs',type=int,default=16);p.add_argument('--batches',type=int,default=1)
    p.add_argument('--minutes',type=float,default=5)
    a=p.parse_args();out=Path(a.output).resolve();ckpath=Path(a.checkpoint).resolve()
    if out.exists():raise FileExistsError('Immutable evaluation output already exists')
    out.mkdir(parents=True)
    started=time.time();deadline=started+a.minutes*60
    write_json(out/'status.json',dict(status='running',pid=os.getpid(),gpu=a.gpu,started_unix=started,deadline_unix=deadline))
    try:
        env,task,_=make_simulator(a.envs,a.gpu,a.seed)
        actor=Actor().to(env.device);critic=Critic(DS+DC).to(env.device)
        ck=torch.load(ckpath,map_location=env.device)
        actor.load_state_dict(ck['learner']['actor']);critic.load_state_dict(ck['learner']['critic'])
        all_episodes=[];parity=[]
        for batch in range(a.batches):
            seed=a.seed+batch
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
            env.reset_buf[:]=1
            env.step(torch.zeros(a.envs,22,device=env.device))
            collector=Collector(env,task,actor,critic)
            first={}
            # An episode can fail on the reset/warm-start frame itself. Preserve
            # it instead of waiting for that environment's next, longer episode.
            for i in env.reset_buf.nonzero().flatten().tolist():
                initial_axis=qrotate(qconj(env.object_init_quat[i:i+1]),env.spin_axis[i:i+1])
                axis=qrotate(env.object_rot[i:i+1],initial_axis)
                tilt=float(torch.acos((axis*env.spin_axis[i:i+1]).sum(-1).clamp(-1,1))[0])
                first[i]=dict(env_id=i,seed=seed,episode=i,failure=True,timeout=False,
                    **{'return':float(env.rew_buf[i]),'rotation':float(env.last_spin_delta_axis[i]/20),
                       'offaxis':float(env.last_spin_delta_offaxis[i]/20),'steps':1,
                       'axis_deviation_rad':tilt,'initial_frame_failure':True})
            while len(first)<a.envs:
                if time.time()>=deadline:raise TimeoutError('Evaluation did not complete within its budget')
                previous=len(collector.completed)
                row=collector.step(noise=0,normalize=False)
                ids=row['native_done'].nonzero().flatten().tolist()
                completed=collector.completed[previous:]
                assert len(ids)==len(completed)
                for env_id,episode in zip(ids,completed):
                    if env_id not in first:first[env_id]=dict(episode,env_id=env_id,seed=seed)
            all_episodes.extend(first[i] for i in range(a.envs));parity.append(collector.evidence())
        report=dict(checkpoint=str(ckpath),checkpoint_sha256=file_digest(ckpath),episodes=all_episodes,
                    episodes_per_seed=a.envs,seeds=list(range(a.seed,a.seed+a.batches)),
                    selection='first complete episode of every environment; no early-completion filtering',
                    mean_return=float(np.mean([e['return'] for e in all_episodes])),
                    mean_rotation=float(np.mean([e['rotation'] for e in all_episodes])),
                    mean_axis_deviation=float(np.mean([e['axis_deviation_rad'] for e in all_episodes])),
                    mean_offaxis=float(np.mean([e['offaxis'] for e in all_episodes])),
                    failures=sum(e['failure'] for e in all_episodes),parity=parity,
                    seconds=time.time()-started)
        write_json(out/'evaluation.json',report)
        import wandb
        wb=wandb.init(entity=os.environ.get('WANDB_ENTITY'),project=os.environ.get('WANDB_PROJECT','unionmimic'),
            name=out.name,job_type='physical-evaluation',dir=str(out),mode=os.environ.get('WANDB_MODE','offline'),
            settings=wandb.Settings(init_timeout=30,disable_git=True,disable_code=True))
        wb.log({k:v for k,v in report.items() if isinstance(v,(float,int))});wb.finish()
        write_json(out/'status.json',dict(status='completed',pid=os.getpid(),gpu=a.gpu,started_unix=started,
            ended_unix=time.time(),deadline_unix=deadline))
        print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','parity']}),flush=True)
    except BaseException as e:
        write_json(out/'status.json',dict(status='failed',pid=os.getpid(),gpu=a.gpu,started_unix=started,
            ended_unix=time.time(),error=repr(e),deadline_unix=deadline));raise


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    finally:
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        sys.stdout.flush();sys.stderr.flush()
    os._exit(code)
