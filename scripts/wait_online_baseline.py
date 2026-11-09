#!/usr/bin/env python3
"""Bounded local queue: wait for the specific predecessor; never stop other jobs."""
import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from online_baseline_lane import ROOT, verify, write
import json


def predecessor_ready(path):
    import json
    lane=Path(path)
    status=json.loads((lane/'lane-status.json').read_text())
    if status['status'] in ('stopped','failed'):
        raise RuntimeError('Predecessor stopped/failed: leaving GPU untouched for diagnosis')
    if status['status'] != 'completed':
        return False
    if status.get('locked_test_pending', True):
        raise RuntimeError('Predecessor completed without its required final evaluation')
    # Its recorded completion is not sufficient if the native child still exists.
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd=entry.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,PermissionError):
            continue
        if str(lane) in cmd and ('train_online' in cmd or 'evaluate_online.py' in cmd or 'online_candidate_lane.py' in cmd):
            return False
    return True


def gpu_idle(gpu):
    memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
        '--format=csv,noheader,nounits']).decode().strip())
    apps=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
        '--format=csv,noheader,nounits']).decode().strip()
    return memory<100 and not apps


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True)
    ap.add_argument('--arm',required=True);a=ap.parse_args()
    plan=verify(a.plan,a.approval);arm=plan['arms'][a.arm]
    out=ROOT/plan['run_root']/'queues'/(a.arm+'.json')
    stopping=[False]
    def stop(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while time.time()<plan['queue_start_deadline_unix'] and not stopping[0]:
            ready=not arm.get('predecessor') or predecessor_ready(ROOT/arm['predecessor'])
            if ready and gpu_idle(arm['gpu']):
                verify(a.plan,a.approval)
                write(out,dict(status='dispatched',gpu=arm['gpu'],unix=time.time(),pid=os.getpid()))
                cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/online_baseline_lane.py'),
                    '--plan',str(Path(a.plan).resolve()),'--approval',str(Path(a.approval).resolve()),
                    '--arm',a.arm,'--run',str(ROOT/plan['run_root']/a.arm)]
                os.execv(cmd[0],cmd)
            write(out,dict(status='waiting',gpu=arm['gpu'],unix=time.time(),pid=os.getpid(),
                           predecessor=arm.get('predecessor'),start_deadline_unix=plan['queue_start_deadline_unix']))
            time.sleep(30)
        write(out,dict(status='stopped' if stopping[0] else 'expired',unix=time.time(),gpu=arm['gpu']))
    except BaseException as exc:
        write(out,dict(status='failed',error=repr(exc),unix=time.time(),gpu=arm['gpu']))
        raise


if __name__=='__main__':main()
