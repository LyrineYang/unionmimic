#!/usr/bin/env python3
"""One-shot, authorized pilot gate. No retries, source edits or recurring watchdog."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from online_baseline_lane import ROOT, write


def audit(plan):
    evidence=ROOT/plan['pilot']['directory']
    records=[]
    for name in plan['pilot']['arms']:
        r=json.loads((evidence/(name+'-acceptance.json')).read_text())
        if not r.get('passed') or not r.get('finite_checkpoint') or not r.get('actor_updated'):
            raise RuntimeError('Pilot failed: '+name)
        if r['counts'] != dict(model=320,actor=12,critic=240,cycles=12):
            raise RuntimeError('Pilot update counts inconsistent: '+name)
        if r['parity']['max_action_error']>1e-6 or r['parity']['max_target_error']>1e-6 or r['parity']['max_reward_error']>2e-3:
            raise RuntimeError('Pilot native parity failed: '+name)
        run=Path(r['run']);status=json.loads((run/'status.json').read_text())
        if status['status']!='segment_complete':raise RuntimeError('Pilot process result incomplete')
        manifest=json.loads((run/'manifest.json').read_text())
        expected=dict(plan['training_config'],**plan['pilot']['overrides'],
                      **plan['arms'][name]['config_overrides'])
        if manifest['config']!=expected or manifest['seed']!=plan['arms'][name]['seed']:
            raise RuntimeError('Pilot configuration does not match this factorial arm: '+name)
        if manifest['released_actor_loaded'] or manifest['prior_loaded']:
            raise RuntimeError('Unexpected policy/prior warm start')
        for path,h in manifest['sources'].items():
            if plan['source_sha256'].get(path)!=h:raise RuntimeError('Pilot source not in accepted plan: '+path)
        records.append(r)
    if len({r['initial_actor_digest'] for r in records})!=1:
        raise RuntimeError('Pilot initialization mismatch')
    for path,h in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=h:raise RuntimeError('Source drift: '+path)
    for file in ('tests.log','coordinator-tests.log'):
        if not (evidence/file).read_text().rstrip().endswith('OK'):
            raise RuntimeError('CPU tests have not passed')
    return dict(passed=True,unix=time.time(),source_sha256=plan['source_sha256'],pilots=records,
                total_pilot_gpu_hours=sum(r['seconds'] for r in records)/3600,
                learning_benefit_proven=False,authorization=plan['authorization'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());base=ROOT/plan['pilot']['directory']
    stopping=[False]
    def stop(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while time.time()<plan['pilot']['gate_deadline_unix'] and not stopping[0]:
            if all((base/(name+'-acceptance.json')).exists() for name in plan['pilot']['arms']):
                result=audit(plan);write(ROOT/plan['acceptance']['evidence'],result)
                cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/launch_online_baseline.py'),
                     '--plan',str(Path(a.plan).resolve()),'--approval',str(Path(a.approval).resolve())]
                with (base/'launch.log').open('w') as log:
                    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
                write(base/'gate-status.json',dict(status='launched',unix=time.time(),session=plan['tmux_session']))
                return
            lines=subprocess.check_output(['tmux','-L','unionmimic','list-panes','-t',plan['pilot']['tmux_session'],
                '-a','-F','#{session_name} #{pane_dead} #{pane_dead_status}']).decode().splitlines()
            for line in lines:
                parts=line.split()
                if parts[0]==plan['pilot']['tmux_session'] and len(parts)>=3 and parts[1]=='1' and parts[2]!='0':
                    raise RuntimeError('Pilot tmux process failed; no formal launch')
            write(base/'gate-status.json',dict(status='waiting_for_pilots',unix=time.time(),
                deadline_unix=plan['pilot']['gate_deadline_unix']))
            time.sleep(30)
        write(base/'gate-status.json',dict(status='stopped' if stopping[0] else 'expired',unix=time.time()))
    except BaseException as exc:
        write(base/'gate-status.json',dict(status='failed',error=repr(exc),unix=time.time()))
        raise


if __name__=='__main__':main()
