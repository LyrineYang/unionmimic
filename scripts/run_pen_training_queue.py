"""One bounded queue, no recurring watchdog, implicit renewal or blind restart."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from unionmimic.common import write_json,file_digest
from unionmimic.training_window import remaining_seconds

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--authorization',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());auth=json.loads(Path(a.authorization).read_text())
    if remaining_seconds(auth)<3600:raise TimeoutError('Less than one hour left; refuse to start a new queue')
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('GPU compute processes already exist; do not interfere or silently oversubscribe')
    run=(ROOT/plan['run']).resolve();data=(ROOT/plan['retarget']).resolve();run.mkdir(parents=True,exist_ok=False)
    write_json(run/'plan.json',plan);write_json(run/'authorization.json',auth)
    active=[];logs=[]
    def status(stage,**kw):write_json(run/'queue-status.json',dict(stage=stage,pid=os.getpid(),unix=time.time(),deadline_unix=auth['deadline_unix'],**kw))
    def launch(script,args,log):
        remaining_seconds(auth);stream=log.open('w');logs.append(stream)
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)]+[str(x) for x in args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        active.append(child);return child
    def wait_all(children,limit):
        until=min(auth['deadline_unix'],time.time()+limit)
        while any(c.poll() is None for c in children):
            if time.time()>=until:raise TimeoutError('Stage budget expired; no automatic retries')
            time.sleep(5)
        if any(c.returncode!=0 for c in children):raise RuntimeError('Stage failed; preserve logs/checkpoints and do not promote')
    def stop(*_):raise InterruptedError('Queue stopped; no automatic restart')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        status('preflight_physics');jobs=[]
        for gpu in range(4):
            args=['--gpu',gpu,'--envs',208,'--seed',20261191+gpu//2,'--retarget',data,'--out',run/f'physics-gpu{gpu}']
            if gpu%2:args+=['--moving']
            jobs.append(launch('check_pen_physics.py',args,run/f'physics-gpu{gpu}.log'))
        wait_all(jobs,600)
        physics=[json.loads((run/f'physics-gpu{i}/result.json').read_text()) for i in range(4)]
        if not all(r['passed'] for r in physics):raise RuntimeError('Held-out hold/moving contact gate failed')
        status('preflight_reference_phases')
        child=launch('check_pen_reference_reset.py',['--gpu',2,'--retarget',data,'--out',run/'reference-phases','--seed',20261193,'--midpoints'],run/'reference-phases.log')
        wait_all([child],300);rsi=json.loads((run/'reference-phases/result.json').read_text())
        if not (rsi['reset_contract_passed'] and rsi['physical_first_step_passed']):raise RuntimeError('Whole-duration midpoint initialization gate failed')
        child=launch('check_pen_reference_reset.py',['--gpu',3,'--retarget',data,'--out',run/'reference-random-phases','--seed',20261203,'--random-phases'],run/'reference-random-phases.log')
        wait_all([child],300);random_rsi=json.loads((run/'reference-random-phases/result.json').read_text())
        if not (random_rsi['reset_contract_passed'] and random_rsi['physical_first_step_passed']):raise RuntimeError('Whole-duration stratified random initialization gate failed')
        status('preflight_learning');jobs=[]
        for lane in plan['lanes']:
            g=lane['gpu'];jobs.append(launch('pilot_pen_learning.py',['--gpu',g,'--algorithm',lane['algorithm'],'--task-set',lane['task_set'],
                '--retarget',data,'--out',run/f'operator-gpu{g}','--minutes',15],run/f'operator-gpu{g}.log'))
        wait_all(jobs,960)
        operators=[json.loads((run/f'operator-gpu{i}/acceptance.json').read_text()) for i in range(4)]
        if not all(x['passed'] and x['used_actor_updates']>=2 for x in operators):raise RuntimeError('Learning/recollection gate failed')
        # Exercise the actual evaluation entry point before launching a long job.
        child=launch('evaluate_pen_policy.py',['--gpu',0,'--checkpoint',run/'operator-gpu0/operator.pt','--retarget',data,
            '--envs',52,'--out',run/'evaluation-smoke.json'],run/'evaluation-smoke.log')
        wait_all([child],600);assert json.loads((run/'evaluation-smoke.json').read_text())['passed']
        gate=dict(passed=True,retarget_manifest_sha256=file_digest(data/'manifest.json'),physics=[x['passed'] for x in physics],
            whole_duration_reset=rsi['physical_first_step_passed'],whole_duration_random_reset=random_rsi['physical_first_step_passed'],operators=[x['passed'] for x in operators],
            learning_gain_verified=False,exact_wuji_reproduction=False)
        write_json(run/'gate.json',gate)
        if remaining_seconds(auth)<8*3600:raise TimeoutError('Not enough of the original window for an 8-hour comparison')
        status('long_training');jobs=[]
        for lane in plan['lanes']:
            g=lane['gpu'];jobs.append(launch('train_pen_comparison.py',['--gpu',g,'--algorithm',lane['algorithm'],'--task-set',lane['task_set'],
                '--retarget',data,'--run',run/f'gpu{g}','--authorization',Path(a.authorization).resolve(),'--gate',run/'gate.json'],run/f'gpu{g}.log'))
        # A failed lane does not kill healthy lanes or trigger a restart.
        until=auth['deadline_unix']
        while any(c.poll() is None for c in jobs) and time.time()<until:
            status('long_training',lanes=[dict(gpu=i,pid=c.pid,returncode=c.poll()) for i,c in enumerate(jobs)])
            time.sleep(min(30,max(.01,until-time.time())))
        status('finished' if all(c.poll()==0 for c in jobs) else 'finished_with_failures_or_deadline',
            lanes=[dict(gpu=i,pid=c.pid,returncode=c.poll()) for i,c in enumerate(jobs)])
    except BaseException as exc:
        status('stopped' if isinstance(exc,InterruptedError) else 'blocked_before_promotion',error=repr(exc));raise
    finally:
        for c in active:
            if c.poll() is None:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        until=time.time()+20
        while any(c.poll() is None for c in active) and time.time()<until:time.sleep(.2)
        for c in active:
            if c.poll() is None:
                try:os.killpg(c.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        for stream in logs:stream.close()
if __name__=='__main__':main()
