"""Bounded unattended loop: SM -> physical evaluation -> collect -> WM calibration."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback
from .common import ROOT, file_digest, write_json
from .telemetry import TrainingMetrics
from .retention import prune_bulk_outputs
from .recovery import STAGES, validate_completed_stage


def run_night(args):
    run=Path(args.run).resolve()
    run.mkdir(parents=True,exist_ok=True)
    previous=None
    resume_phase=None
    first_cycle=1
    if args.resume_run:
        previous=json.loads((run/'status.json').read_text())
        assert previous['status']!='running', 'Refusing to duplicate a running supervisor'
        assert previous['phase'] in STAGES, 'This stage needs manual review before recovery'
        first_cycle=previous['cycle']
        resume_phase=previous['phase']
        write_json(run/('status-before-resume-%d.json'%(previous.get('resume_count',0)+1)),previous)
    else:
        assert not (run/'status.json').exists(), 'Use --resume-run for an interrupted run'
    started=previous['started_unix'] if previous else time.time()
    deadline=started+args.hours*3600
    baseline=json.loads(Path(args.baseline).read_text())
    current=Path(previous['checkpoint'] if previous else args.checkpoint).resolve()
    cache=Path(previous['cache'] if previous else args.cache).resolve()
    assert current.is_file() and cache.is_file()
    original_data=Path(args.data).resolve()
    recent=[]
    actor_lr=args.lr
    anchor_coef=args.anchor_coef
    state=dict(pid=os.getpid(),status='running',started_unix=started,
               checkpoint=str(current),cache=str(cache),cycle=0,phase='initializing')
    if previous:
        state.update({k:v for k,v in previous.items() if k not in ('error','traceback','child_pid','command','log')})
        state.update(pid=os.getpid(),status='running',resumed_unix=time.time(),
                     resume_count=previous.get('resume_count',0)+1)
        actor_lr=previous.get('effective_actor_lr',args.lr)
        anchor_coef=previous.get('effective_anchor_coefficient',args.anchor_coef)
        best=json.loads((run/'best-policy.json').read_text())
        recent=[p/'data' for p in sorted(run.glob('cycle-*'))
                if int(p.name.split('-')[-1])<first_cycle and
                (p/'data/episodes.h5').is_file() and (p/'data/collection.json').is_file()][-3:]
    else:
        best=dict(checkpoint=baseline['checkpoint'],evaluation=baseline,
                  note='Best measured policy; training may continue from a lower-return candidate within the regression bound.')
        write_json(run/'best-policy.json',best)
    state['deadline_unix']=deadline
    child=None
    def stopping(signum,frame):
        if child is not None and child.poll() is None:
            child.terminate()
        state.update(status='stopped',stopped_unix=time.time())
        write_json(run/'status.json',state)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stopping)
    signal.signal(signal.SIGINT,stopping)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    suffix='-resume-%d'%state['resume_count'] if previous else ''
    write_json(run/('config'+suffix+'.json'),configuration)
    metrics=TrainingMetrics(run,configuration,previous)
    if not previous:
        metrics.emit('evaluation',0,dict(mean_return=baseline['mean_return'],
                     baseline_return=baseline['mean_return'],failure_rate=baseline['failure_rate']))
    sources=list((ROOT/'unionmimic').glob('*.py'))+list((ROOT/'patches').glob('*.patch'))
    write_json(run/('source-manifest'+suffix+'.json'),{str(p.relative_to(ROOT)):file_digest(p) for p in sources})
    write_json(run/'status.json',state)
    def execute(phase,params,folder):
        nonlocal child
        remaining=deadline-time.time()
        if remaining<=0:
            raise TimeoutError('Configured overnight time budget reached')
        recovering=previous is not None and state['cycle']==first_cycle
        if recovering and STAGES.index(phase)<=STAGES.index(resume_phase):
            proof=validate_completed_stage(phase,params)
            write_json(folder/(phase+suffix+'-verified.json'),proof)
            metrics.emit('recovery',state['cycle'],{phase+'_reused':1})
            print(json.dumps(dict(event='recovered_stage',**proof)),flush=True)
            return
        logfile=folder/(phase+(suffix if recovering else '')+'.log')
        phase_started=time.monotonic()
        metrics.begin_phase(phase,state['cycle'])
        command=[sys.executable,'-m','unionmimic.cli']+[str(v) for v in params]
        with open(logfile,'w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            state.update(phase=phase,child_pid=child.pid,log=str(logfile),command=command,
                         checkpoint=str(current),cache=str(cache),updated_unix=time.time())
            write_json(run/'status.json',state)
            print(json.dumps(state),flush=True)
            try:
                with open(logfile) as reader:
                    heartbeat=0.
                    while True:
                        while True:
                            offset=reader.tell()
                            line=reader.readline()
                            if not line or not line.endswith('\n'):
                                reader.seek(offset)
                                break
                            try:
                                row=json.loads(line)
                            except ValueError:
                                continue
                            if isinstance(row,dict):
                                metrics.ingest(row,phase,state['cycle'])
                        code=child.poll()
                        if code is not None:
                            # Drain output written between the last read and exit.
                            for line in reader:
                                try:
                                    row=json.loads(line)
                                except ValueError:
                                    continue
                                if isinstance(row,dict):
                                    metrics.ingest(row,phase,state['cycle'])
                            break
                        if time.time()>=deadline:
                            child.terminate()
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            raise TimeoutError('Configured overnight time budget reached')
                        metrics.sample_gpu(args.train_gpu,state['cycle'])
                        if time.monotonic()-heartbeat>10:
                            heartbeat=time.monotonic()
                            state.update(updated_unix=time.time(),attempted_updates=dict(metrics.totals))
                            write_json(run/'status.json',state)
                        time.sleep(.5)
            finally:
                state.pop('child_pid',None)
            metrics.emit('runtime',state['cycle'],{phase+'_seconds':time.monotonic()-phase_started})
            if code:
                raise RuntimeError('%s exited with code %s; see %s' % (phase,code,logfile))
    try:
        for cycle in range(first_cycle,args.cycles+1):
            if time.time()>=deadline:
                break
            if shutil.disk_usage(run).free < 50*1024**3:
                state.update(status='disk_budget_reached',phase='stopped_before_next_cycle')
                break
            state['cycle']=cycle
            folder=run/('cycle-%03d'%cycle)
            folder.mkdir(exist_ok=previous is not None and cycle==first_cycle)
            import torch
            loaded=torch.load(str(current),map_location='cpu')
            resume=['--resume'] if 'sm_state' in loaded else []
            del loaded
            execute('sm',['sm','--cache',cache,'--checkpoint',current,'--output',folder/'sm',
                          '--gpu',args.train_gpu,'--updates',args.sm_updates,'--seed',args.seed+cycle,
                          '--lr',actor_lr,'--anchor-coef',anchor_coef]+resume+
                          (['--fork-seed'] if args.fork_seed else []),folder)
            candidate=folder/'sm/last.pt'
            execute('evaluate',['evaluate','--checkpoint',candidate,'--output',folder/'evaluation.json',
                                 '--gpu',args.sim_gpu,'--envs',8,'--seed',123],folder)
            result=json.loads((folder/'evaluation.json').read_text())
            return_floor=baseline['mean_return']-args.regression_tolerance*max(abs(baseline['mean_return']),1)
            accepted=(result['mean_return']>=return_floor and
                      result['failure_rate']<=baseline['failure_rate']+.125)
            reuse_decision=(previous is not None and cycle==first_cycle and
                            STAGES.index(resume_phase)>STAGES.index('evaluate'))
            if reuse_decision:
                decision=json.loads((folder/'decision.json').read_text())
                assert decision['evaluation']['checkpoint_sha256']==result['checkpoint_sha256']
                assert decision['accepted']==accepted
                current=Path(decision['candidate'] if accepted else decision['previous_checkpoint'])
            else:
                write_json(folder/'decision.json',dict(accepted=accepted,evaluation=result,
                           baseline=baseline,return_floor=return_floor,
                           previous_checkpoint=str(current),candidate=str(candidate)))
            if accepted:
                current=candidate
            elif not reuse_decision:
                actor_lr=max(actor_lr*.5,1e-7)
                anchor_coef=min(anchor_coef*1.5,100.)
            if not reuse_decision:
                metrics.emit('evaluation',cycle,dict(mean_return=result['mean_return'],
                             baseline_return=baseline['mean_return'],return_ratio=result['mean_return']/baseline['mean_return'],
                             failure_rate=result['failure_rate'],accepted=accepted,
                             next_actor_lr=actor_lr,next_anchor_coefficient=anchor_coef))
            if (result['mean_return']>best['evaluation']['mean_return'] and
                    result['failure_rate']<=best['evaluation']['failure_rate']):
                shutil.copyfile(candidate,run/'best-policy.pt.tmp')
                os.replace(run/'best-policy.pt.tmp',run/'best-policy.pt')
                best=dict(checkpoint=str(run/'best-policy.pt'),evaluation=result)
                write_json(run/'best-policy.json',best)
            state['best_policy_checkpoint']=best['checkpoint']
            state['best_mean_return']=best['evaluation']['mean_return']
            state['last_evaluation']=result
            state['last_candidate_accepted']=accepted
            state.update(effective_actor_lr=actor_lr,effective_anchor_coefficient=anchor_coef)
            if cycle==args.cycles or deadline-time.time()<180:
                break
            data=folder/'data'
            execute('collect',['collect','--checkpoint',current,'--output',data,'--gpu',args.sim_gpu,
                               '--envs',16,'--steps',args.collect_steps,'--probe-noise',.25,
                               '--seed',args.seed+1000+cycle],folder)
            recent=(recent+[data])[-3:]
            collection=json.loads((data/'collection.json').read_text())
            metrics.emit('data',cycle,{k:collection[k] for k in ('episodes','transitions','failures','falls','seconds')})
            dataset=folder/'dataset'
            execute('merge',['merge','--data',original_data,*recent,'--output',dataset],folder)
            prepared=folder/'prepared'
            execute('prepare',['prepare','--data',dataset,'--output',prepared,
                               '--gpu',args.train_gpu,'--checkpoint',current],folder)
            new_cache=prepared/'cache.pt'
            calibrated=folder/'calibrated'
            execute('calibrate',['calibrate','--cache',new_cache,'--checkpoint',current,
                                 '--output',calibrated,'--gpu',args.train_gpu,'--resume',
                                 '--updates',args.calibration_updates],folder)
            report=json.loads((calibrated/'validation.json').read_text())
            if not report['passed']:
                # Continue actual model training before allowing another actor update.
                execute('calibrate-extend',['calibrate','--cache',new_cache,
                    '--checkpoint',calibrated/'last.pt','--output',folder/'calibrated-extended',
                    '--gpu',args.train_gpu,'--resume','--updates',3*args.calibration_updates],folder)
                calibrated=folder/'calibrated-extended'
                report=json.loads((calibrated/'validation.json').read_text())
            if not report['passed']:
                state.update(status='calibration_gate_failed',phase='stopped_before_actor_update',
                             calibration_report=str(calibrated/'validation.json'))
                break
            current=calibrated/'best.pt'
            cache=new_cache
            freed=prune_bulk_outputs(run,cycle,args.keep_cycles,
                   [current,cache,best['checkpoint']]+[p/'episodes.h5' for p in recent])
            metrics.emit('retention',cycle,dict(freed_gib=freed/1024**3))
        else:
            state['status']='completed'
        if state['status']=='running':
            state['status']='completed'
    except TimeoutError as exc:
        state.update(status='time_budget_reached',message=str(exc))
    except Exception as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        state.update(checkpoint=str(current),cache=str(cache),updated_unix=time.time())
        state['attempted_updates']=dict(metrics.totals)
        write_json(run/'status.json',state)
        metrics.close()
        print(json.dumps(state),flush=True)
