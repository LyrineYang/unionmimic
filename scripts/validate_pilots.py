"""One-shot post-pilot checks. Never starts training or a recurring watchdog."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import time
from pathlib import Path
from unionmimic.common import ROOT,write_json,file_digest


def await_complete(run, arm):
    deadline=time.monotonic()+3000
    while time.monotonic()<deadline:
        path=run/arm/'status.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state['status']=='failed':raise RuntimeError(arm+' failed: '+state.get('error',''))
            if state['status']=='completed':return state
        time.sleep(10)
    raise TimeoutError('Pilot completion not observed within 50 minutes: '+arm)


def check_arm(run,arm,gpu):
    folder=run/'validation'/arm;folder.mkdir(parents=True,exist_ok=True)
    def execute(name,command,timeout=900,allow_gate_failure=False):
        with open(folder/(name+'.log'),'x') as log:
            r=subprocess.run([sys.executable]+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if r.returncode and not allow_gate_failure:raise RuntimeError(name+' failed; see '+str(folder/(name+'.log')))
        return r.returncode
    try:
        await_complete(run,arm)
        ck=run/arm/'last.pt';source_sha=file_digest(ck)
        m=json.loads((run/'launch.json').read_text())
        prediction=None
        if arm=='dynamics':
            execute('rebuild',['-m','unionmimic.cli','prepare','--data',str(ROOT/m['dataset']),
                '--checkpoint',str(ck),'--output',str(folder/'prepared'),'--gpu',str(gpu),'--recompute-state'])
            code=execute('prediction',['-m','unionmimic.cli','validate','--cache',str(folder/'prepared/cache.pt'),
                '--checkpoint',str(ck),'--output',str(folder/'prediction.json'),'--gpu',str(gpu)],allow_gate_failure=True)
            # A documented rejection is a result; missing/partial output is an error.
            prediction=json.loads((folder/'prediction.json').read_text())
            if code and prediction.get('passed') is not False:raise RuntimeError('Incomplete prediction validation')
        evaluations=[];baseline=[]
        for seed in m['evaluation']['periodic_dev_seeds']:
            name='dev-%d'%seed
            execute(name,['scripts/diagnose_reward.py','--checkpoint',str(ck),'--output',str(folder/(name+'.json')),
                '--gpu',str(gpu),'--seed',str(seed),'--save-starts'])
            evaluations.append(json.loads((folder/(name+'.json')).read_text()))
            baseline.append(json.loads((ROOT/'work/analysis/pretraining-review-20261101'/('baseline-dev-%d.json'%seed)).read_text()))
        assert file_digest(ck)==source_sha
        mean=lambda rows,key:sum(r[key] for r in rows)/len(rows)
        old=mean(baseline,'mean_return');new=mean(evaluations,'mean_return')
        old_rotation=sum(r['mean_components']['rotation_radians'] for r in baseline)/len(baseline)
        new_rotation=sum(r['mean_components']['rotation_radians'] for r in evaluations)/len(evaluations)
        failures=sum(r['failures'] for r in evaluations)
        gate=(new>=.95*old and new_rotation>=.95*old_rotation and failures==0)
        report=dict(status='validated',arm=arm,checkpoint=str(ck),sha256=source_sha,
            episodes=sum(r['episodes'] for r in evaluations),mean_return=new,baseline_return=old,
            return_change_percent=100*(new/old-1),rotation_radians=new_rotation,
            baseline_rotation_radians=old_rotation,failures=failures,
            prediction_passed=prediction['passed'] if prediction else None,
            pilot_stability_gate=gate and (prediction is None or prediction['passed']),
            statistical_improvement_claim=False,locked_test_opened=False,
            long_training_started=False,next='Review pilot stability, expanded SM direction test and paired-seed allocation before long training')
        write_json(folder/'result.json',report)
        return report
    except BaseException as exc:
        report=dict(status='validation_error',arm=arm,error=repr(exc),long_training_started=False)
        write_json(folder/'result.json',report)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(check_arm,run,arm,gpu) for arm,gpu in [('ppo',1),('dynamics',2)]]
        results=[f.result() for f in futures]
    write_json(run/'validation-summary.json',dict(results=results,long_training_started=False))
    if any(r['status']=='validation_error' for r in results):sys.exit(1)
