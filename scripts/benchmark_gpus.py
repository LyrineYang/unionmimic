#!/usr/bin/env python3
"""Measure four simultaneous physics/evaluation workers before a long run."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
output=a.output.resolve()
output.mkdir(parents=True,exist_ok=True)
jobs=[]
for gpu in range(4):
    target=output/('gpu-%d.json'%gpu)
    assert not target.exists()
    log=open(output/('gpu-%d.log'%gpu),'w')
    command=['bash',str(root/'scripts/run_training.sh'),'evaluate','--gpu',str(gpu),
             '--envs','8','--seed','123','--checkpoint',str(a.checkpoint.resolve()),'--output',str(target)]
    began=time.monotonic()
    child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    jobs.append((gpu,child,log,began))
results=[]
while jobs:
    for job in list(jobs):
        gpu,child,log,began=job
        code=child.poll()
        if code is None:
            continue
        log.close()
        row=dict(gpu=gpu,exit_code=code,wall_seconds=time.monotonic()-began)
        if code==0:
            result=json.loads((output/('gpu-%d.json'%gpu)).read_text())
            row.update(episodes=result['episodes'],mean_return=result['mean_return'],failures=result['failures'])
        results.append(row)
        jobs.remove(job)
        print(json.dumps(row),flush=True)
    if jobs:
        time.sleep(1)
(output/'benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['exit_code']==0 for r in results)
