"""Assemble completed control evaluations and exploratory probes, without selection."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run=Path(a.run).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[];rng=np.random.default_rng(817)
    for arm in ('none','fixed','rollout'):
        lane=run/arm;result=json.loads((lane/'result.json').read_text());status=json.loads((lane/'status.json').read_text())
        assert result['passed'] and status['status']=='completed' and result['updates']==400
        assert json.loads((lane/'evaluation-completed.json').read_text())['completed']
        evs={key:json.loads((lane/('eval-'+key+'.json')).read_text()) for key in ('initial','clean','pulse')}
        row=dict(arm=arm,training=result,evaluation={})
        for key,ev in evs.items():
            assert ev['passed'] and ev['episodes']==128
            values=np.array([x['return'] for x in ev['rows']]);inds=rng.integers(0,len(values),(5000,len(values)))
            row['evaluation'][key]={k:ev[k] for k in ('mean_return','mean_rotation_rad','failure_rate','mean_axis_deviation_rad','mean_offaxis_rad','mean_steps','seconds','checkpoint_sha256')}
            row['evaluation'][key]['episode_bootstrap_return95']=np.quantile(values[inds].mean(1),[.025,.975]).tolist()
        rows.append(row)
    record=dict(rows=rows,scope='10Hz diagnostic with 256 camera environments. Fresh learners, 300 warmup + 100 policy updates.',
        limits=['One training seed and one evaluation seed; environment CIs are not independent training-seed confidence.',
                'All arms include a shadow frozen observer; total timing cannot compare to camera-free SM.',
                'Fixed/rollout priors have extra effective input capacity. No attribution to pretraining without more controls.',
                '100 actor updates are an early-learning screen, not a converged baseline or long-training result.'])
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8));colors=['#4c78a8','#f58518','#54a24b'];labels=['SM / no context','Fixed start context','Frozen prior rollout']
    for i,(row,color,label) in enumerate(zip(rows,colors,labels)):
        ev=row['evaluation'];axs[0,0].bar(i-.17,ev['clean']['mean_return'],width=.32,color=color)
        axs[0,0].bar(i+.17,ev['pulse']['mean_return'],width=.32,color=color,alpha=.45,hatch='//')
        axs[0,1].bar(i-.17,ev['clean']['failure_rate']*100,width=.32,color=color)
        axs[0,1].bar(i+.17,ev['pulse']['failure_rate']*100,width=.32,color=color,alpha=.45,hatch='//')
        events=[json.loads(l) for l in (run/row['arm']/'events.jsonl').read_text().splitlines()]
        events=[e for e in events if e['kind']=='update']
        axs[1,0].plot([e['update'] for e in events],[e['wm_pos_mm'] for e in events],color=color,label=label)
        actor=[e for e in events if 'imag_r' in e]
        axs[1,1].plot([e['update']-300 for e in actor],[e['imag_r'] for e in actor],color=color,label=label)
    axs[0,0].set_title('Independent evaluation: raw episode return')
    axs[0,1].set_title('Failure rate (%)')
    for ax in axs[0]:ax.set_xticks(range(3));ax.set_xticklabels(['SM','Fixed','Rollout']);ax.grid(axis='y',alpha=.2)
    axs[1,0].set_title('Model fit diagnostic: training-batch position error')
    axs[1,0].set_ylabel('mm');axs[1,0].set_xlabel('Model-fit rounds')
    axs[1,1].set_title('Imagined reward (not simulator return)');axs[1,1].set_xlabel('Actor updates')
    axs[1,1].legend(fontsize=8)
    fig.suptitle('UnionMimic: frozen-context early-learning experiment',fontsize=16)
    fig.text(.5,.015,'Solid: clean evaluation. Hatched: command-noise pulse. 128 initial environments per arm; one training seed.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(out/'control-results.png',dpi=160);fig.savefig(out/'control-results.svg')
    print(json.dumps([{k:v for k,v in r.items() if k!='training'} for r in rows]),flush=True)
if __name__=='__main__':main()
