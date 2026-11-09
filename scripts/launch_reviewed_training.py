"""Launch the gated follow-up in fresh tmux windows; never restart the old run."""
import argparse,json,shlex,shutil,subprocess,sys,time,socket
from pathlib import Path
from unionmimic.common import ROOT,file_digest,write_json
from unionmimic.comparison import budget


def main(a):
    evidence=a.evidence.resolve();gates=json.loads((evidence/'launch-gates.json').read_text())
    assert gates['ppo_ready'] and gates['code_review_passed']
    for path,sha in gates['evidence_sha256'].items():assert file_digest(path)==sha,path
    arm='sm' if gates['sm_ready'] else 'dynamics'
    if arm=='dynamics':assert gates['dynamics_ready']
    ledger=json.loads((evidence/'resource-ledger.json').read_text())
    lane_hours={0:10.6,1:9.8,2:9.8,3:9.8}
    baseline=json.loads((evidence/'baseline-summary.json').read_text())
    initial=ROOT/'work/runs/step2-20261101'
    ppo=json.loads((evidence/'ppo-recheck/result.json').read_text());assert ppo['passed'] and ppo['resume_verified']
    run=a.run.resolve();assert not run.exists(),'Use a new run; existing runs are immutable'
    configuration=dict(started_unix=time.time(),experiment=('paired_ppo_sm' if arm=='sm' else 'ppo_headroom_and_rssm_action_coverage'),
        reason=gates['decision'],checkpoint=str(initial/'calibrated/best.pt'),
        dataset=str(initial/'dataset'),cache=str(initial/'prepared-v2/cache.pt'),
        prior_gpu_hours=ledger['total_gpu_hours'],baseline=baseline,
        baseline_prediction_score=gates['baseline_prediction_score'],
        ppo=dict(envs=256,batch_frames=16384,actor_lr=2e-5),
        sm=dict(actor_lr=3e-6,anchor_coefficient=10.,horizon=4,batch=128),
        dynamics=dict(lr=1e-5,batch=8,length=32,horizon=4,feature_drift_coefficient=.1,probe_noise=.5),
        lanes=[dict(gpu=g,arm='ppo' if g in (0,2) else arm,
                    seed=(51001+g//2 if g in (0,2) or arm=='sm' else 63001+g//2),
                    train_hours=8.,total_hours=lane_hours[g],prior_card_hours=ledger['per_card_hours'][str(g)]) for g in range(4)],
        development_hours=[.5,1,2,4,6,7.7],periodic_dev_seeds=list(range(71001,71005)),
        full_dev_seeds=list(range(71001,71017)),test_seeds=list(range(81001,81026)),
        engineering_targets=dict(return_improvement_fraction=.05,signed_rotation_non_decreasing=True,
             failure_noninferiority_margin=.02,model_h4_rmse_reduction_fraction=.1),
        wandb='disabled',recurring_watchdog=False,socket='unionmimic',session='unionmimic-followup',tensorboard_port=6008,
        evaluation_note='Best selected on periodic development; confirm128 development episodes, then locked200. Physical episodes are not additional training seeds.',
        implementation_limit='Full online RSSM+SM alternation is not implemented; failed SM direction checks cannot be bypassed by forward prediction metrics.')
    budget(configuration)
    occupied=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    if occupied:raise RuntimeError('GPU compute processes still exist; inspect ownership before launching: '+occupied)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',6008))
    sources=list((ROOT/'unionmimic').glob('*.py'))
    sources += [ROOT/'scripts'/name for name in ('train_ppo_segment.py','train_dynamics_segment.py','train_comparison.py',
                'launch_reviewed_training.py','diagnose_reward.py','env.sh')]
    old=json.loads((ROOT/'work/runs/pilot-20261101-v1/launch.json').read_text())
    sources += [ROOT/p for p in old['source_sha256'] if p.startswith('work/upstreams/')]
    configuration['source_sha256']={str(p.relative_to(ROOT)):file_digest(p) for p in sorted(set(sources))}
    inputs=[initial/'calibrated/best.pt',initial/'dataset/episodes.h5',initial/'prepared-v2/cache.pt',
            evidence/'launch-gates.json',evidence/'resource-ledger.json']
    configuration['immutable_inputs']=[dict(path=str(p),sha256=file_digest(p)) for p in inputs]
    # Refuse an existing tmux session before creating a run or training process.
    probe=subprocess.run(['tmux','-L','unionmimic','has-session','-t',configuration['session']],capture_output=True)
    if probe.returncode==0:raise RuntimeError('Target tmux session already exists')
    run.mkdir(parents=True);write_json(run/'launch.json',configuration)
    write_json(run/'approval.json',dict(approved=True,arms=['ppo',arm],manifest_sha256=file_digest(run/'launch.json'),
        authorization='User explicitly requested result-based long-plan optimization, four-card budget verification and actual training launch. Passing evidence is bound in launch-gates.json.'))
    for p in sources:
        target=run/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shutil.copyfile(evidence/'launch-gates.json',run/'acceptance.json');shutil.copyfile(evidence/'resource-ledger.json',run/'resource-ledger.json')
    commands=[]
    for g in range(4):
        lane=run/('gpu%d'%g);lane.mkdir()
        command=[sys.executable,'scripts/train_comparison.py','--manifest',str(run/'launch.json'),
                 '--approval',str(run/'approval.json'),'--gpu',str(g)]
        script='source scripts/env.sh\n'+shlex.join(command)+' > '+shlex.quote(str(lane/'supervisor.log'))+' 2>&1'
        if g==0:
            tmux=['tmux','-L','unionmimic','new-session','-d','-s',configuration['session'],'-n','gpu0','-c',str(ROOT),'bash','-lc',script]
        else:
            tmux=['tmux','-L','unionmimic','new-window','-t',configuration['session'],'-n','gpu%d'%g,'-c',str(ROOT),'bash','-lc',script]
        subprocess.run(tmux,check=True);commands.append(dict(gpu=g,command=command))
        if g==0:subprocess.run(['tmux','-L','unionmimic','set-option','-t',configuration['session'],'remain-on-exit','on'],check=True)
    script='source scripts/env.sh\nexec tensorboard --logdir '+shlex.quote(str(run))+' --host 127.0.0.1 --port 6008'
    subprocess.run(['tmux','-L','unionmimic','new-window','-t',configuration['session'],'-n','curves','-c',str(ROOT),'bash','-lc',script],check=True)
    write_json(run/'started.json',dict(unix=time.time(),commands=commands,session=configuration['session'],wandb=False,watchdog=False))
    print(json.dumps(dict(run=str(run),allocation=configuration['lanes'],maximum_gpu_hours=ledger['total_gpu_hours']+sum(lane_hours.values()))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    main(p.parse_args())
