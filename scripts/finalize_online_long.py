#!/usr/bin/env python3
"""After both approved lanes finish, evaluate locked tests serially on one free GPU."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from online_long_lane import ROOT,verify,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True)
    a=p.parse_args();plan=verify(a.plan,a.approval);root=ROOT/plan['run_root']
    root.mkdir(parents=True,exist_ok=True)
    waiting_until=time.time()+9*3600
    while time.time()<waiting_until:
        states=[]
        for arm in plan['arms']:
            path=root/arm/'lane-status.json'
            states.append(json.loads(path.read_text()) if path.exists() else {'status':'pending'})
        if any(s['status'] in ['failed','stopped'] for s in states):
            write(root/'final-status.json',dict(status='blocked',reason='Training failed or was stopped; no automatic restart/test.'));return
        if all(s['status']=='completed' for s in states):break
        time.sleep(30)
    else:
        write(root/'final-status.json',dict(status='blocked',reason='Training completion not observed within nine hours.'));return
    verify(a.plan,a.approval)
    gpu=plan['locked_test']['gpu'];reports={}
    for arm in plan['arms']:
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits']).decode())
        if used>=100:
            write(root/'final-status.json',dict(status='blocked',reason='Evaluation GPU occupied; other tasks preserved.',gpu=gpu));return
        lane=root/arm;ck=lane/'best-by-development-return.pt'
        if not ck.exists():
            write(root/'final-status.json',dict(status='blocked',reason='No development-selected checkpoint',arm=arm));return
        out=lane/'locked-test';minutes=plan['locked_test']['hours_each_max']*60
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/evaluate_online.py'),'--checkpoint',str(ck),
             '--output',str(out),'--gpu',str(gpu),'--envs','20','--batches','10',
             '--seed',str(plan['locked_test']['seed']),'--minutes',str(minutes)]
        with (lane/'locked-test.log').open('w') as log:
            result=subprocess.run(['timeout','--signal=TERM','--kill-after=20s',str(int(minutes*60+20))+'s']+cmd,
                                  stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        if result.returncode:
            write(root/'final-status.json',dict(status='failed',arm=arm,reason='Locked evaluation incomplete; inspect log.'));return
        reports[arm]=str(out/'evaluation.json')
    write(root/'final-status.json',dict(status='completed',reports=reports,ended_unix=time.time(),
        note='One training seed per method. Report failures/precision and full measured cost; no paper-scale speedup claim.'))

if __name__=='__main__':main()
