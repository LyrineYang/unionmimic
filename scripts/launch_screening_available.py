#!/usr/bin/env python3
"""Launch reviewed arms on currently idle GPUs without taking another user's GPU."""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
from screening_lane import ROOT, verify
from screening_summary import summarize
from wait_online_baseline import gpu_idle


def validate_queues(plan, now=None):
    now=time.time() if now is None else now
    queues=plan['gpu_queues'];names=[n for q in queues.values() for n in q]
    if not queues or not set(queues).issubset({'0','1','2','3'}):raise ValueError('Invalid GPU queue')
    if len(names)!=8 or len(set(names))!=8 or set(names)!=set(plan['arms']):raise ValueError('Require all eight unique arms')
    if not 0<plan['arm_minutes']<=90:raise ValueError('Arm exceeds reviewed screening budget')
    if plan['segments']*plan['segment_minutes']+plan['segments']*plan['evaluation']['minutes']>=plan['arm_minutes']:
        raise ValueError('Insufficient startup/evaluation reserve')
    if now+max(map(len,queues.values()))*plan['arm_minutes']*60>plan['hard_stop_unix']:
        raise ValueError('Queues cannot finish before hard deadline')
    if plan['authorization'].get('automatic_long_training',False):raise ValueError('No automatic long training')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    path=Path(a.plan).resolve();plan=verify(path);validate_queues(plan)
    for f,h in plan['orchestrator_source_sha256'].items():
        if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h:raise RuntimeError('Orchestrator source drift')
    pilot=summarize(ROOT/plan['pilot_plan'])
    evidence=json.loads((ROOT/plan['pilot_acceptance']).read_text())
    if not pilot['engineering_passed'] or not evidence['engineering_passed']:
        raise RuntimeError('Pilot acceptance failed/incomplete')
    if pilot['source_sha256']!=plan['source_sha256'] or evidence['source_sha256']!=plan['source_sha256']:
        raise RuntimeError('Pilot/plan training sources differ')
    if len(evidence.get('checkpoint_checks',[]))!=8 or not all(x['finite'] and x['exact_counts_and_frames'] for x in evidence['checkpoint_checks']):
        raise RuntimeError('Checkpoint verification missing')
    if a.validate_only:print('Available-GPU plan validated');return
    root=ROOT/plan['run_root']
    if root.exists():raise FileExistsError(root)
    for gpu in plan['gpu_queues']:
        if not gpu_idle(int(gpu)):raise RuntimeError('GPU occupied: '+gpu)
    session=plan['tmux_session']
    if subprocess.run(['tmux','-L','unionmimic','has-session','-t','='+session],stderr=subprocess.DEVNULL).returncode==0:raise RuntimeError('Session exists')
    root.mkdir(parents=True);(root/'plan.json').write_bytes(path.read_bytes())
    for i,gpu in enumerate(plan['gpu_queues']):
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/screening_lane.py'),'--plan',str(path),'--gpu',gpu]
        shell='source scripts/env.sh && export WANDB_MODE=online && exec '+shlex.join(cmd)
        tm=['tmux','-L','unionmimic','new-session' if i==0 else 'new-window','-d']
        tm+=['-s',session] if i==0 else ['-t',session]
        subprocess.run(tm+['-n','gpu'+gpu,'-c',str(ROOT),shell],check=True)
    subprocess.run(['tmux','-L','unionmimic','set-option','-t',session,'remain-on-exit','on'],check=True)
    receipt=dict(launched=True,session=session,gpus=list(map(int,plan['gpu_queues'])),started_unix=time.time(),
        estimated_latest_end_unix=time.time()+max(map(len,plan['gpu_queues'].values()))*plan['arm_minutes']*60,
        hard_stop_unix=plan['hard_stop_unix'],pilot_gpu_hours=pilot['actual_gpu_hours'])
    (root/'launch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
