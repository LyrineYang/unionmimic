"""Plot existing measurements without extrapolating incomplete PPO learning."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'work/analysis/sm-ppo-available-20261109-v1'
    out.mkdir(parents=True,exist_ok=True)
    sm=json.loads((ROOT/'work/analysis/sm-attainment-20261104-v1/results.json').read_text())
    ppo=[json.loads(line) for line in (ROOT/'work/runs/sharpa-paired-ppo-12h-20261104-v1/events.jsonl').read_text().splitlines()
         if json.loads(line).get('kind')=='evaluation']
    rows=[]
    keys=['mean_return','mean_rotation_rad','failure_rate','mean_axis_deviation_rad']
    for method,source in [('SM',sm),('PPO-Sim',ppo)]:
        for r in source:
            assert r['passed'] and r['episodes']==512 and r['seed']==20261105
            rows.append(dict(method=method,minutes=r['checkpoint_elapsed_seconds']/60,
                             **{k:r[k] for k in keys},episodes=r['episodes'],seed=r['seed'],
                             checkpoint_sha256=r['checkpoint_sha256']))
    result=dict(rows=rows,speedup_established=False,training_seeds=1,
                evaluation='512 first completed episodes per checkpoint; fixed development seed, all failures included.',
                timing='SM early checkpoint times use historical file timestamps; PPO uses saved event elapsed time. Different evaluation schedules.',
                conclusion='SM achieves effective rotation within approximately two hours. PPO coverage is insufficient for time-to-matched-performance comparison.',
                new_heldout_evaluation=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'measurements.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    labels=['Return / episode','Rotation / episode (rad)','Failure rate (%)','Mean axis deviation (rad)']
    for ax,key,label in zip(axes.flat,keys,labels):
        for method,color in [('SM','#18827c'),('PPO-Sim','#c36b22')]:
            subset=[r for r in rows if r['method']==method];mult=100 if key=='failure_rate' else 1
            ax.plot([r['minutes'] for r in subset],[r[key]*mult for r in subset],
                    'o-' if method=='SM' else 's',color=color,label=method+' measured',markersize=6)
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xlim(0,125)
    for ax in axes[1]:ax.set_xlabel('Training wall time at checkpoint (minutes)')
    axes[0,0].legend();fig.suptitle('Sharpa single-object SM learning: existing measured checkpoints',fontsize=14)
    fig.text(.5,.025,'PPO has only one early measurement. No speedup or extrapolated PPO convergence time is claimed.',ha='center',fontsize=10,color='#8c3c14')
    fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(out/'learning-coverage.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
