#!/usr/bin/env python3
"""Launch the authorized eight-arm screening only, in four tmux lanes."""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
from screening_lane import ROOT, verify
from wait_online_baseline import gpu_idle


def validate(plan):
    queues=plan['gpu_queues'];flat=[x for q in queues.values() for x in q]
    if set(queues)!={'0','1','2','3'} or len(flat)!=8 or len(set(flat))!=8 or set(flat)!=set(plan['arms']):
        raise ValueError('Require eight unique arms on four GPU queues')
    if any(len(q)!=2 for q in queues.values()) or not 0<plan['arm_minutes']<=90:
        raise ValueError('Screening exceeds reviewed budget')
    if plan['segments']*plan['segment_minutes']+plan['segments']*plan['evaluation']['minutes']>=plan['arm_minutes']:
        raise ValueError('Insufficient startup and checkpoint reserve')
    if time.time()+2*plan['arm_minutes']*60>plan['hard_stop_unix']:
        raise ValueError('Two waves cannot finish before hard stop')
    if plan['authorization'].get('automatic_long_training',False):
        raise ValueError('This entry cannot authorize future long training')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    path=Path(a.plan).resolve();plan=verify(path);validate(plan)
    if plan.get('pilot_plan'):
        from screening_summary import summarize
        pilot=summarize(ROOT/plan['pilot_plan'])
        if not pilot['complete'] or not pilot['engineering_passed']:
            raise RuntimeError('All eight engineering pilots must pass')
        if pilot['source_sha256']!=plan['source_sha256']:
            raise RuntimeError('Pilot sources differ from screening sources')
    if a.validate_only:print('Validated screening plan');return
    root=ROOT/plan['run_root']
    if root.exists():raise FileExistsError(root)
    for gpu in range(4):
        if not gpu_idle(gpu):raise RuntimeError('GPU occupied: '+str(gpu))
    session=plan['tmux_session']
    if subprocess.run(['tmux','-L','unionmimic','has-session','-t',session],stderr=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('tmux session exists')
    root.mkdir(parents=True);(root/'plan.json').write_bytes(path.read_bytes())
    for gpu in range(4):
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/screening_lane.py'),'--plan',str(path),'--gpu',str(gpu)]
        shell='source scripts/env.sh && export WANDB_MODE=online && exec '+shlex.join(cmd)
        tm=['tmux','-L','unionmimic','new-session' if gpu==0 else 'new-window','-d']
        tm+=['-s',session] if gpu==0 else ['-t',session]
        subprocess.run(tm+['-n','gpu%d'%gpu,'-c',str(ROOT),shell],check=True)
    subprocess.run(['tmux','-L','unionmimic','set-option','-t',session,'remain-on-exit','on'],check=True)
    print(json.dumps(dict(launched=True,session=session,started_unix=time.time(),hard_stop_unix=plan['hard_stop_unix'])))

if __name__=='__main__':main()
