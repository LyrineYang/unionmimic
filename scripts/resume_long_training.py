#!/usr/bin/env python3
"""Resume failed tmux workers after validating their committed stage outputs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from unionmimic.common import ROOT,write_json
from unionmimic.recovery import validate_completed_stage

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--gpus',type=int,nargs='+',required=True)
a=p.parse_args()
run=a.run.resolve()
launch=json.loads((run/'launch.json').read_text())
cfg=launch['configuration']
tmux=['tmux','-L',cfg['socket']]
prepared=[]
for gpu in a.gpus:
    folder=run/('gpu%d'%gpu)
    state=json.loads((folder/'status.json').read_text())
    assert state['status']=='failed', 'Only failed workers are recovered by this command'
    assert time.time()<state['started_unix']+cfg['hours']*3600, 'Original time budget has expired'
    target=cfg['session']+':gpu%d'%gpu
    dead=subprocess.check_output(tmux+['display-message','-p','-t',target,'#{pane_dead}'],text=True).strip()
    assert dead=='1', 'Refusing to replace a live tmux pane'
    params=state['command'][3:]
    proof=validate_completed_stage(state['phase'],params)
    command=list(launch['commands'][cfg['gpus'].index(gpu)])+['--resume-run']
    assert str(folder) in command
    prepared.append((gpu,target,command,proof))
for gpu,target,command,proof in prepared:
    folder=run/('gpu%d'%gpu)
    write_json(folder/('recovery-preflight-%d.json'%int(time.time())),proof)
    subprocess.run(tmux+['respawn-pane','-t',target,'-c',str(ROOT),shlex.join(command)],check=True)
    print(json.dumps(dict(gpu=gpu,target=target,recovered=proof)),flush=True)
