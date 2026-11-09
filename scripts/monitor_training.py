#!/usr/bin/env python3
"""Read-only tmux overview of four workers; no torch or credentials required."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
a=p.parse_args()
while True:
    print('\033[2J\033[HUnionMimic: four independent seeds, 12-hour budget',flush=True)
    print('GPU  STATUS                      CYCLE PHASE          SM UPDATES  LAST RETURN  BEST RETURN')
    finished=0
    for gpu in range(4):
        try:
            s=json.loads((a.run/('gpu%d'%gpu)/'status.json').read_text())
        except (FileNotFoundError,ValueError):
            print('%3d  starting'%gpu)
            continue
        last=s.get('last_evaluation',{}).get('mean_return',float('nan'))
        best=s.get('best_mean_return',float('nan'))
        count=s.get('attempted_updates',{}).get('sm',0)
        print('%3d  %-27s %5d %-14s %10d %12.2f %12.2f'%(gpu,s['status'],s['cycle'],s['phase'],count,last,best))
        finished+=s['status']!='running'
    print('\nTensorBoard: http://127.0.0.1:6006 (SSH port forwarding for remote access)\n')
    subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw',
                    '--format=csv,noheader'],check=False)
    print('\nUpdated:',time.strftime('%Y-%m-%d %H:%M:%S %Z'),flush=True)
    if finished==4:
        break
    time.sleep(5)
