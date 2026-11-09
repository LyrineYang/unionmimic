"""Read-only run report. Development evaluations are not a locked test set."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();run=Path(a.run).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];manifest=json.loads((run/'manifest.json').read_text())
    events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines() if x.strip()]
    updates=[e for e in events if e['kind']=='update'];evals=[]
    for e in events:
        if e['kind']!='evaluation':continue
        f=run/(e['name']+'.json');d=json.loads(f.read_text());rows=d.pop('rows')
        d.update(name=e['name'],hours=(e['unix']-manifest['started_unix'])/3600,
                 failure_count=sum(x['failure'] for x in rows),mean_steps=float(np.mean([x['steps'] for x in rows])),
                 offaxis_rad=float(np.mean([x['offaxis_rad'] for x in rows])))
        evals.append(d)
    changed=[p for p,h in manifest['sources'].items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    result=dict(checked_unix=time.time(),run=str(run),status=json.loads((run/'status.json').read_text()),
        elapsed_hours=(time.time()-manifest['started_unix'])/3600,evaluations=evals,
        best=json.loads((run/'best-evaluation.json').read_text()),latest=updates[-1],source_changes=changed,
        comparable_ppo_completed=False,protocol='512 fixed development episodes; not external benchmark or multi-seed result')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(10,6.5))
    hrs=np.array([r['hours'] for r in evals])
    for ax,key,label in zip(axs.flat,['mean_return','mean_rotation_rad','failure_rate','mean_axis_deviation_rad'],
                            ['Native return / episode','Rotation / episode (rad)','Failure fraction','Mean axis deviation (rad)']):
        ax.plot(hrs,[r[key] for r in evals],'o-');ax.set(xlabel='Elapsed hours (includes warm-up)',ylabel=label);ax.grid(alpha=.25)
    fig.suptitle('Sharpa + pinned LEAP SM: development evaluations (512 episodes)')
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('status','best','source_changes')},indent=2))

if __name__=='__main__':main()
