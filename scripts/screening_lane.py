#!/usr/bin/env python3
"""Two bounded screening arms on one GPU; no long-run launch or silent recovery."""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from online_baseline_lane import ROOT, write
from wait_online_baseline import gpu_idle


def verify(path):
    p=json.loads(Path(path).read_text())
    if p.get('authorization',{}).get('screening_approved') is not True:
        raise PermissionError('Screening has no explicit user authorization')
    for name,h in p['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:
            raise RuntimeError('Source drift: '+name)
    if hashlib.sha256(Path(p['parent_checkpoint']).read_bytes()).hexdigest()!=p['parent_sha256']:
        raise RuntimeError('Parent checkpoint changed')
    return p


def validate_evaluation(path, checkpoint, episodes):
    e=json.loads((path/'evaluation.json').read_text())
    s=json.loads((path/'status.json').read_text())
    if s['status']!='completed' or len(e['episodes'])!=episodes:
        raise RuntimeError('Incomplete development evaluation')
    if e['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise RuntimeError('Evaluation checkpoint mismatch')
    if len({(x['seed'],x['env_id']) for x in e['episodes']})!=episodes:
        raise RuntimeError('Evaluation contains duplicate episodes')
    return {k:v for k,v in e.items() if k not in ('episodes','parity')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--gpu',type=int,required=True)
    a=ap.parse_args();plan=verify(a.plan);gpu=a.gpu
    root=ROOT/plan['run_root'];root.mkdir(parents=True,exist_ok=True)
    status_file=root/('gpu%d.json'%gpu)
    stopped=[False];child=[None];started=time.time()
    def stop(*_):
        stopped[0]=True
        if child[0] and child[0].poll() is None:os.killpg(child[0].pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(cmd, log, deadline):
        if stopped[0]:raise InterruptedError('User stopped screening')
        if time.time()>=deadline:raise TimeoutError('Screening deadline expired')
        with open(log,'a') as stream:
            child[0]=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child[0].wait(timeout=deadline-time.time())
            except subprocess.TimeoutExpired:
                os.killpg(child[0].pid,signal.SIGTERM)
                try:child[0].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child[0].pid,signal.SIGKILL);child[0].wait()
                raise TimeoutError('Stage exceeded bounded screening budget')
        if stopped[0]:raise InterruptedError('User stopped screening')
        if code:raise RuntimeError('Stage failed with exit %d; see %s'%(code,log))
    py=str(ROOT/'.venv/bin/python')
    completed=[]
    try:
        for arm_id in plan['gpu_queues'][str(gpu)]:
            verify(a.plan)
            if stopped[0]:raise InterruptedError('User stopped screening')
            if time.time()>=plan['latest_arm_start_unix']:raise TimeoutError('Too late to start another arm before review')
            if not gpu_idle(gpu):raise RuntimeError('GPU occupied: leaving other processes untouched')
            arm=plan['arms'][arm_id];dest=root/arm_id
            if dest.exists():raise FileExistsError('Immutable arm already exists: '+arm_id)
            dest.mkdir();begin=time.time();deadline=min(begin+plan['arm_minutes']*60,plan['hard_stop_unix'])
            cfg={**plan['training_config'],**arm,'deadline_unix':deadline,
                 'branch_from':plan['parent_checkpoint'],'branch_sha256':plan['parent_sha256'],
                 'branch_initial_cycles':plan['parent_counts']['cycles']}
            write(dest/'config.json',cfg)
            write(dest/'launch.json',dict(started_unix=begin,deadline_unix=deadline,gpu=gpu,arm=arm_id,
                plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),parent_sha256=plan['parent_sha256']))
            results=[]
            for segment in range(plan['segments']):
                verify(a.plan)
                write(status_file,dict(status='running',phase='train',arm=arm_id,gpu=gpu,pid=os.getpid(),segment=segment,
                    started_unix=started,updated_unix=time.time(),deadline_unix=deadline,completed=completed))
                cmd=[py,str(ROOT/'scripts/train_online_screen.py'),'--run',str(dest/'learner'),'--arm','sm',
                     '--gpu',str(gpu),'--seed',str(plan['seed']),'--minutes',str(plan['arm_minutes']),
                     '--cycles',str(plan.get('cycles_per_segment',10**9)),'--segment-minutes',str(plan['segment_minutes']),
                     '--config',str(dest/'config.json')]
                if segment:cmd.append('--resume')
                # Reserve the final evaluation time; never silently skip it.
                run(cmd,dest/('train-%d.log'%segment),deadline-plan['evaluation']['minutes']*60)
                state=json.loads((dest/'learner/status.json').read_text())
                if state['status']!='segment_complete':raise RuntimeError('Worker did not complete clean segment: '+state['status'])
                ck=dest/('segment-%d.pt'%segment);shutil.copy2(dest/'learner/last.pt',ck)
                evdir=dest/('development-%d'%segment)
                write(status_file,dict(status='running',phase='evaluate',arm=arm_id,gpu=gpu,pid=os.getpid(),segment=segment,
                    started_unix=started,updated_unix=time.time(),deadline_unix=deadline,completed=completed))
                ev=plan['evaluation']
                run([py,str(ROOT/'scripts/evaluate_online.py'),'--checkpoint',str(ck),'--output',str(evdir),
                     '--gpu',str(gpu),'--seed',str(ev['seed']),'--envs',str(ev['envs']),
                     '--batches',str(ev['batches']),'--minutes',str(ev['minutes'])],dest/('evaluate-%d.log'%segment),deadline)
                results.append(validate_evaluation(evdir,ck,ev['envs']*ev['batches']))
            write(dest/'summary.json',dict(status='completed',arm=arm_id,gpu=gpu,started_unix=begin,ended_unix=time.time(),
                deadline_unix=deadline,actual_gpu_hours=(time.time()-begin)/3600,development=results,
                parent_sha256=plan['parent_sha256'],from_scratch=False,counts=state['counts'],frames=state['frames']))
            completed.append(arm_id)
        write(status_file,dict(status='completed',completed=completed,gpu=gpu,pid=os.getpid(),started_unix=started,ended_unix=time.time()))
    except BaseException as exc:
        stop()
        if child[0] and child[0].poll() is None:
            try:child[0].wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child[0].pid,signal.SIGKILL);child[0].wait()
        write(status_file,dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc),
            gpu=gpu,pid=os.getpid(),started_unix=started,ended_unix=time.time(),completed=completed))
        raise

if __name__=='__main__':main()
