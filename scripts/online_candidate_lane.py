#!/usr/bin/env python3
"""One approved lane: training/dev segments then a locked test within one deadline.

The launch approval is deliberately external; this module never approves itself.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def read_plan(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    for path,h in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=h:raise RuntimeError('Source drift: '+path)
    if not plan['acceptance']['engineering_passed']:raise RuntimeError('Pilot acceptance not passed')
    return plan

def verify(plan_path,approval_path):
    raw=Path(plan_path).read_bytes()
    approval=json.loads(Path(approval_path).read_text())
    if not approval.get('approved') or approval.get('plan_sha256')!=hashlib.sha256(raw).hexdigest():
        raise PermissionError('Long training awaits the user confirmation of this exact plan.')
    return read_plan(plan_path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True)
    ap.add_argument('--arm',required=True);ap.add_argument('--run',required=True)
    a=ap.parse_args();plan=verify(a.plan,a.approval);arm=plan['arms'][a.arm]
    run=Path(a.run).resolve()
    if run.exists():raise FileExistsError('Long-run directories are immutable; diagnose before any resume')
    run.mkdir(parents=True)
    started=time.time();deadline=started+arm['hours']*3600
    training_deadline=deadline-plan['locked_test']['reserve_seconds']
    algorithm=arm['algorithm'];seed=arm['seed']
    cfg={**plan['training_config'],**arm.get('config_overrides',{}),'deadline_unix':training_deadline,
         'launch_authorization':dict(approval_path=str(Path(a.approval).resolve()),
             plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest())}
    write(run/'config.json',cfg)
    write(run/'launch.json',dict(plan=str(Path(a.plan).resolve()),started_unix=started,
        deadline_unix=deadline,training_deadline_unix=training_deadline,gpu=arm['gpu'],arm=a.arm,
        algorithm=algorithm,seed=seed,approval_sha256=hashlib.sha256(Path(a.approval).read_bytes()).hexdigest()))
    current=[None];stopping=[False]
    def stop(*_):
        stopping[0]=True
        if current[0] and current[0].poll() is None:os.killpg(current[0].pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def execute(cmd,log_path,until=None):
        remaining=(training_deadline if until is None else until)-time.time()
        if remaining<=0:return False
        with open(log_path,'a') as f:
            current[0]=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
            try:code=current[0].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(current[0].pid,signal.SIGTERM)
                try:current[0].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(current[0].pid,signal.SIGKILL);current[0].wait()
                return False
        if stopping[0]:return False
        if code:raise RuntimeError('Child exited %d; inspect %s'%(code,log_path))
        return True
    python=str(ROOT/'.venv/bin/python');segment=0;best=None;final_segment=False
    try:
        while time.time()<training_deadline and not stopping[0]:
            verify(a.plan,a.approval)
            write(run/'lane-status.json',dict(status='running',phase='train',pid=os.getpid(),segment=segment,
                started_unix=started,deadline_unix=deadline,training_deadline_unix=training_deadline,updated_unix=time.time()))
            cmd=[python,str(ROOT/'scripts/train_online_candidate.py'),'--run',str(run/'learner'),'--arm',algorithm,
                 '--gpu',str(arm['gpu']),'--seed',str(seed),'--minutes',str(arm['hours']*60),
                 '--cycles',str(10**9 if final_segment else arm['cycles_per_segment']),'--config',str(run/'config.json')]
            if segment:cmd.append('--resume')
            if not execute(cmd,run/('train-%02d.log'%segment)):break
            state=json.loads((run/'learner/status.json').read_text())
            if state['status']=='completed':break
            if state['status']!='segment_complete':raise RuntimeError('Unexpected worker exit '+state['status'])
            # Evaluation consumes the same lane budget. No overlapping
            # training process and no optimistic exclusion of camera/launch cost.
            if training_deadline-time.time()<plan['development']['reserve_seconds']:
                segment+=1;final_segment=True;continue
            ckpt=run/'checkpoints'/('segment-%02d.pt'%segment);ckpt.parent.mkdir(exist_ok=True)
            shutil.copy2(run/'learner/last.pt',ckpt)
            dest=run/('development-%02d'%segment)
            cmd=[python,str(ROOT/'scripts/evaluate_online.py'),'--checkpoint',str(ckpt),'--output',str(dest),
                 '--gpu',str(arm['gpu']),'--envs',str(plan['development']['envs']),
                 '--batches',str(plan['development']['batches']),'--seed',str(plan['development']['seed']),
                 '--minutes',str(plan['development']['reserve_seconds']/60)]
            if not execute(cmd,run/('evaluate-%02d.log'%segment)):break
            ev=json.loads((dest/'evaluation.json').read_text())
            if best is None or ev['mean_return']>best['mean_return']:
                best=dict(mean_return=ev['mean_return'],mean_rotation=ev['mean_rotation'],failures=ev['failures'],
                          evaluation=str(dest/'evaluation.json'),checkpoint=str(ckpt))
                shutil.copy2(ckpt,run/'best-by-development-return.pt');write(run/'best.json',best)
            segment+=1
        final_report=None
        if not stopping[0]:
            verify(a.plan,a.approval)
            if best is None:raise RuntimeError('No development-selected checkpoint; locked test cannot proceed')
            write(run/'lane-status.json',dict(status='running',phase='locked_test',pid=os.getpid(),
                started_unix=started,deadline_unix=deadline,training_deadline_unix=training_deadline,updated_unix=time.time()))
            dest=run/'locked-test'
            cmd=[python,str(ROOT/'scripts/evaluate_online.py'),'--checkpoint',str(run/'best-by-development-return.pt'),
                 '--output',str(dest),'--gpu',str(arm['gpu']),'--envs',str(plan['locked_test']['envs']),
                 '--batches',str(plan['locked_test']['batches']),'--seed',str(plan['locked_test']['seed']),
                 '--minutes',str(max(0,deadline-time.time())/60)]
            if not execute(cmd,run/'locked-test.log',until=deadline):
                if not stopping[0]:raise RuntimeError('Locked test exceeded the original lane deadline; no extension')
            else:
                final_report=str(dest/'evaluation.json')
                ev=json.loads(Path(final_report).read_text())
                if len(ev['episodes'])!=plan['locked_test']['episodes_each']:
                    raise RuntimeError('Locked test episode count incomplete')
        write(run/'lane-status.json',dict(status='stopped' if stopping[0] else 'completed',phase='finished',
            pid=os.getpid(),started_unix=started,deadline_unix=deadline,ended_unix=time.time(),
            best=best,locked_test_pending=final_report is None,final_report=final_report))
    except BaseException as e:
        stop();write(run/'lane-status.json',dict(status='failed',phase='exception',error=repr(e),
            started_unix=started,deadline_unix=deadline,ended_unix=time.time()));raise

if __name__=='__main__':main()
