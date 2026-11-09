#!/usr/bin/env python3
"""Launch the measured four-worker schedule under a dedicated tmux server."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--config',type=Path,default=Path('config/long-training.json'))
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
run=a.run.resolve(); source=a.source.resolve()
cfg=json.loads(a.config.read_text())
benchmark=json.loads((run/'benchmark/benchmark.json').read_text())
assert len(benchmark)==4 and all(r['exit_code']==0 for r in benchmark)
assert shutil.disk_usage(run).free>150*1024**3, 'Need 150 GiB free for this bounded schedule'
assert all(not (run/('gpu%d'%gpu)).exists() for gpu in cfg['gpus'])
with socket.socket() as s:
    s.bind(('127.0.0.1',cfg['tensorboard_port']))
tmux=['tmux','-L',cfg['socket']]
assert subprocess.run(tmux+['has-session','-t',cfg['session']],capture_output=True).returncode!=0
memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
used=[float(v) for v in memory.splitlines()]
assert all(used[gpu]<1024 for gpu in cfg['gpus']), 'A selected GPU is already occupied'
checkpoint=source/'calibrated/best.pt'
assert checkpoint.is_file() and (source/'prepared-v2/cache.pt').is_file()
sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
for gpu in cfg['gpus']:
    baseline=json.loads((run/('benchmark/gpu-%d.json'%gpu)).read_text())
    assert baseline['checkpoint_sha256']==sha
commands=[]
for gpu,seed in zip(cfg['gpus'],cfg['seeds']):
    command=['bash',str(root/'scripts/run_lane.sh'),str(run/('gpu%d'%gpu)),
        '--data',str(source/'dataset'),'--cache',str(source/'prepared-v2/cache.pt'),
        '--checkpoint',str(checkpoint),'--baseline',str(run/('benchmark/gpu-%d.json'%gpu)),
        '--train-gpu',str(gpu),'--sim-gpu',str(gpu),'--seed',str(seed),'--hours',str(cfg['hours']),
        '--cycles','1000','--sm-updates',str(cfg['sm_updates_per_cycle']),
        '--calibration-updates',str(cfg['calibration_updates_per_cycle']),
        '--collect-steps',str(cfg['collection_steps_per_cycle']),
        '--lr',str(cfg['actor_learning_rate']),'--anchor-coef',str(cfg['anchor_coefficient']),
        '--regression-tolerance',str(cfg['regression_tolerance']),
        '--keep-cycles',str(cfg['keep_recent_cycles']),'--fork-seed']
    commands.append(command)
estimate=dict(basis='Four concurrent 8-episode evaluations and measured single-worker training/data timings',
              measured_evaluation_seconds=[r['wall_seconds'] for r in benchmark],
              estimated_cycles_per_gpu=[45,70],estimated_total_actor_updates=[90000,140000],
              estimated_total_complete_transitions=[1400000,2200000],
              estimated_peak_disk_gib=[80,120],gpu_hour_budget=48,
              note='Ranges are estimates, not promised throughput; calibration and evaluation gates may stop a worker.')
(run/'estimate.json').write_text(json.dumps(estimate,indent=2)+'\n')
launch=dict(started_unix=time.time(),configuration=cfg,source_checkpoint_sha256=sha,commands=commands)
(run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
for gpu,command in zip(cfg['gpus'],commands):
    if gpu==cfg['gpus'][0]:
        args=['new-session','-d','-s',cfg['session'],'-n','gpu%d'%gpu,'-x','160','-y','40']
    else:
        args=['new-window','-t',cfg['session'],'-n','gpu%d'%gpu]
    subprocess.run(tmux+args+['-c',str(root),shlex.join(command)],check=True)
subprocess.run(tmux+['set-option','-g','remain-on-exit','on'],check=True)
for name,command in [('monitor',['python3',str(root/'scripts/monitor_training.py'),str(run)]),
                     ('curves',['bash',str(root/'scripts/run_tensorboard.sh'),str(run)])]:
    subprocess.run(tmux+['new-window','-t',cfg['session'],'-n',name,'-c',str(root),shlex.join(command)],check=True)
subprocess.run(tmux+['select-window','-t',cfg['session']+':monitor'],check=True)
print(json.dumps(dict(session=cfg['session'],socket=cfg['socket'],run=str(run),estimate=estimate),indent=2))
