#!/usr/bin/env python3
"""Launch an explicitly authorized, hash-locked four-arm baseline experiment."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from online_baseline_lane import ROOT, verify
from wait_online_baseline import gpu_idle


def validate_budget(plan):
    arms=plan['arms']
    if len(arms)!=4 or len({x['gpu'] for x in arms.values()})!=4:
        raise ValueError('Exactly four distinct GPU lanes required')
    if sum(x['hours'] for x in arms.values())>36 or any(not 8.5<=x['hours']<=12 for x in arms.values()):
        raise ValueError('Experiment budget exceeded')
    for x in arms.values():
        if x['hours']*3600-plan['locked_test']['reserve_seconds']<8*3600:
            raise ValueError('Less than eight training/development hours')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    plan_path=Path(a.plan).resolve();approval=Path(a.approval).resolve()
    plan=verify(plan_path,approval)
    validate_budget(plan)
    arms=plan['arms']
    # verify() already checked the actual pilot evidence, not a prospective flag.
    if time.time()>=plan['queue_start_deadline_unix']:
        raise RuntimeError('Queue start deadline expired')
    commands=[]
    for name, arm in arms.items():
        if (ROOT/plan['run_root']/name).exists():raise FileExistsError(name)
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/wait_online_baseline.py'),
             '--plan',str(plan_path),'--approval',str(approval),'--arm',name]
        shell='source scripts/env.sh && python scripts/check_runtime.py && export WANDB_MODE=online && exec '+shlex.join(cmd)
        commands.append((name,arm['gpu'],shell))
    if a.validate_only:
        print(json.dumps(dict(validated=True,commands=commands),indent=2));return
    for name,arm in arms.items():
        if not arm.get('predecessor') and not gpu_idle(arm['gpu']):
            raise RuntimeError('GPU occupied; no process will be stopped: '+name)
    session=plan['tmux_session']
    if subprocess.run(['tmux','-L','unionmimic','has-session','-t',session],stderr=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('Session already exists')
    for i,(name,gpu,shell) in enumerate(commands):
        cmd=['tmux','-L','unionmimic','new-session' if i==0 else 'new-window','-d']
        cmd+=['-s',session] if i==0 else ['-t',session]
        subprocess.run(cmd+['-n',name,'-c',str(ROOT),shell],check=True)
    subprocess.run(['tmux','-L','unionmimic','set-option','-t',session,'remain-on-exit','on'],check=True)
    print('Launched '+session)


if __name__=='__main__':main()
