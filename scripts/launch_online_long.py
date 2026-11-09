#!/usr/bin/env python3
"""Reviewable tmux launcher, gated by the user's requested final confirmation."""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from online_long_lane import ROOT,verify,read_plan

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    plan_path=Path(a.plan).resolve();approval=Path(a.approval).resolve()
    plan=read_plan(plan_path) if a.validate_only else verify(plan_path,approval)
    assert len(plan['arms'])==4
    assert len({x['gpu'] for x in plan['arms'].values()})==4
    assert all(8<=x['hours']<=12 for x in plan['arms'].values())
    assert sum(x['hours'] for x in plan['arms'].values())<=48
    assert plan['locked_test']['reserve_seconds']>=0
    assert all(x['hours']*3600-plan['locked_test']['reserve_seconds']>=8*3600 for x in plan['arms'].values())
    assert plan['budget']['authorization']=='user_all_idle_gpus_up_to_12h_20261102'
    commands=[]
    for arm,cfg in plan['arms'].items():
        run=ROOT/plan['run_root']/arm
        assert not run.exists(),str(run)
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/online_long_lane.py'),
             '--plan',str(plan_path),'--approval',str(approval),'--arm',arm,'--run',str(run)]
        shell='source scripts/env.sh && python scripts/check_runtime.py && export WANDB_MODE=online && exec '+ ' '.join(shlex.quote(x) for x in cmd)
        commands.append((arm,cfg['gpu'],shell))
    if a.validate_only:
        print(json.dumps(dict(validated=True,requires_user_confirmation=True,commands=commands),indent=2));return
    for _,gpu,_ in commands:
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits']).decode().strip())
        if used>=100:raise RuntimeError('GPU%d is occupied; no other process will be stopped'%gpu)
    session=plan['tmux_session']
    if subprocess.run(['tmux','-L','unionmimic','has-session','-t',session],stderr=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('tmux session already exists')
    for i,(arm,_,shell) in enumerate(commands):
        tmux=['tmux','-L','unionmimic','new-session' if i==0 else 'new-window','-d']
        tmux+=['-s',session] if i==0 else ['-t',session]
        tmux+=['-n',arm,'-c',str(ROOT),shell]
        subprocess.run(tmux,check=True)
    subprocess.run(['tmux','-L','unionmimic','set-option','-t',session,'remain-on-exit','on'],check=True)
    print('Launched '+session)

if __name__=='__main__':main()
