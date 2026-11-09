#!/usr/bin/env python3
"""Read local progress; rank repeated DEVELOPMENT evidence only. Never launch."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def summarize(plan_path):
    p=json.loads(Path(plan_path).read_text());root=ROOT/p['run_root'];arms={};passed=True
    for name in p['arms']:
        d=root/name
        if not (d/'summary.json').exists():
            arms[name]={'status':'pending'}
            if (d/'learner/status.json').exists():
                s=json.loads((d/'learner/status.json').read_text());arms[name].update(s)
            continue
        s=json.loads((d/'summary.json').read_text());ev=s['development'];ck=d/'learner/last.pt'
        count=s['counts']['cycles']-p['parent_counts']['cycles']
        if count<=0 or len(ev)!=p['segments']:passed=False
        manifest=json.loads((d/'learner/manifest.json').read_text())
        if manifest['branch_sha256']!=p['parent_sha256']:passed=False
        branch=[];cycles=[]
        for line in (d/'learner/events.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['kind']=='branch':branch.append(row)
            if row['kind']=='cycle':cycles.append(row)
        resumes=[json.loads(l) for l in (d/'learner/events.jsonl').read_text().splitlines() if json.loads(l)['kind']=='resume']
        if not branch or len(resumes)!=p['segments']-1 or not all(x.get('curriculum_restored') for x in resumes):passed=False
        if cycles and any(not math.isfinite(float(x[k])) for x in cycles for k in ['actor_grad','model_loss','critic_loss']):passed=False
        ac=json.loads((d/'learner/acceptance.json').read_text())
        parity=ac['parity']
        if parity['max_action_error']>1e-6 or parity['max_target_error']>1e-6 or parity['max_reward_error']>1e-4:passed=False
        if branch and branch[0]['actor_digest']==ac['final_actor_digest']:passed=False
        arms[name]=dict(status='completed',cycles=count,gpu=s['gpu'],actual_gpu_hours=s['actual_gpu_hours'],
            evaluations=ev,actor_start=branch[0]['actor_digest'] if branch else None,
            mean_return=sum(x['mean_return'] for x in ev)/len(ev),
            mean_rotation=sum(x['mean_rotation'] for x in ev)/len(ev),
            mean_axis_deviation=sum(x['mean_axis_deviation'] for x in ev)/len(ev),
            failure_rate=sum(x['failures'] for x in ev)/(len(ev)*p['evaluation']['envs']*p['evaluation']['batches']),
            parity=parity,checkpoint=str(ck),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
            median_cycle_seconds=sorted(x['seconds'] for x in cycles)[len(cycles)//2] if cycles else None)
    complete=all(x['status']=='completed' for x in arms.values())
    if complete and len({x['actor_start'] for x in arms.values()})!=1:passed=False
    ranking=[]
    if complete:
        # Repeated dev seeds are paired measurements, NOT independent 64-episode evidence.
        ranking=sorted(arms,key=lambda n:(arms[n]['failure_rate']<=.05 and arms[n]['mean_rotation']>=28.20587593317032,
            arms[n]['mean_return']),reverse=True)
    return dict(checked_unix=time.time(),complete=complete,engineering_passed=complete and passed,
        arms=arms,ranking=ranking,selection='mean of two fixed development checkpoints; joint gate first, then return',
        repeated_development_seeds_not_independent=True,source_sha256=p['source_sha256'],
        actual_gpu_hours=sum(x.get('actual_gpu_hours',0) for x in arms.values()),
        long_training_launched=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output');a=ap.parse_args()
    result=summarize(a.plan)
    if a.output:Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
