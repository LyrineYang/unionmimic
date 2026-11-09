#!/usr/bin/env python
"""Bounded adversarial reset test; altered fall distance is test-only."""
import ctypes,os,sys,traceback
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
import torch
import time
from pathlib import Path
from unionmimic.online.simulator import make_simulator,Collector
from unionmimic.online.core import Actor,Critic
from unionmimic.online.task import DS,DC
from unionmimic.common import write_json

def main():
    output=Path(sys.argv[1]).resolve();gpu=int(sys.argv[2]);started=time.time()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',dict(status='running',gpu=gpu,pid=os.getpid(),started_unix=started))
    try:
        env,task,_=make_simulator(2,gpu,95101)
        actor=Actor().to(env.device);critic=Critic(DS+DC).to(env.device)
        collector=Collector(env,task,actor,critic)
        original=env.fall_dist
        env.fall_dist=0.;task.cfg['fall_dist']=0.;env.reset_buf[:]=1
        for _ in range(8):
            row=collector.step(noise=0)
            assert not row['valid'].any(), 'Reset transitions entered learner replay'
            assert row['native_done'].all() and row['failure'].all()
        assert len(collector.completed)==16
        assert len({e['episode'] for e in collector.completed})==16
        assert all(e['failure'] and e['steps']==1 for e in collector.completed)
        env.fall_dist=original;task.cfg['fall_dist']=original
        for _ in range(20):collector.step(noise=.1)
        write_json(output/'result.json',dict(passed=True,first_frame_failures_counted=16,
            reset_edges_excluded=True,parity=collector.evidence(),temporary_test_override_restored=True))
        write_json(output/'status.json',dict(status='completed',gpu=gpu,pid=os.getpid(),started_unix=started,ended_unix=time.time()))
    except BaseException as e:
        write_json(output/'status.json',dict(status='failed',gpu=gpu,pid=os.getpid(),started_unix=started,ended_unix=time.time(),error=repr(e)));raise

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    finally:
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        sys.stdout.flush();sys.stderr.flush()
    os._exit(code)
