#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import signal

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('--stop',action='store_true')
a=p.parse_args()
run=a.run.resolve()
state=json.loads((run/'status.json').read_text())
pid=state['pid']
try:
    command=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
except FileNotFoundError:
    command=''
live='unionmimic.cli night' in command and str(run) in command
if a.stop:
    if not live:
        raise SystemExit('The recorded training process is no longer running; no signal sent.')
    # The supervisor terminates its child. Its process group may belong to a
    # tmux pane, so signal only the verified training process.
    os.kill(pid,signal.SIGTERM)
    state['stop_requested']=True
state['process_live']=live
print(json.dumps(state,indent=2))
