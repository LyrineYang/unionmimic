"""Four independent, budgeted PPO/SM lanes with immutable launch configuration.

Resume reuses validated, hashed stage receipts; incomplete stages are preserved
and require explicit diagnosis. No timer, automatic retry, or budget extension.
"""
import fcntl,json,math,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from .common import ROOT,file_digest,write_json


def read(path):return json.loads(Path(path).read_text())


def budget(plan):
    assert plan['prior_gpu_hours']>=0
    lanes=plan['lanes']
    assert len(lanes)==4 and len({x['gpu'] for x in lanes})==4
    for lane in lanes:
        assert 8<=lane['train_hours']<=lane['total_hours']<=12
        assert lane['total_hours']+lane['prior_card_hours']<=12
    assert sum(x['total_hours'] for x in lanes)+plan['prior_gpu_hours']<=48


def deadlines(plan,lane):
    budget(plan)
    return (plan['started_unix']+3600*lane['train_hours'],
            plan['started_unix']+3600*lane['total_hours'])


def aggregate(rows):
    assert rows and len({r['checkpoint_sha256'] for r in rows})==1
    assert len({r['seed'] for r in rows})==len(rows)
    values=[v for r in rows for v in r['returns']]
    assert values and all(math.isfinite(v) for v in values)
    return dict(mean_return=sum(values)/len(values),episodes=len(values),
        rotation_radians=sum(sum(r['episode_components']['rotation_radians']) for r in rows)/len(values),
        failures=sum(r['failures'] for r in rows),checkpoint_sha256=rows[0]['checkpoint_sha256'],
        seeds=[r['seed'] for r in rows],returns=values)


def stable(result,baseline):
    return (result['mean_return']>=.95*baseline['mean_return'] and
            result['rotation_radians']>=.95*baseline['rotation_radians'] and
            result['failures']==0)


def verify_receipt(path,command):
    r=read(path)
    if r['command']!=command:raise ValueError('Stage command changed on resume')
    for name,sha in r['artifacts'].items():
        if file_digest(name)!=sha:raise ValueError('Committed artifact changed: '+name)
    return r


def expected_worker_deadline(code,log,now,deadline):
    endings=('TimeoutError: No PPO update completed within the remaining budget',
             'TimeoutError: No RSSM checkpoint before original deadline',
             'TimeoutError: Original training deadline has passed')
    return code==1 and now>=deadline and log.rstrip().endswith(endings)


def run(args):
    entered=time.time()
    from .review_gate import ReviewGate
    from .recovery import validate_completed_stage
    from .retention import prune_bulk_outputs
    manifest=args.manifest.resolve();plan=read(manifest);budget(plan)
    lane=next(x for x in plan['lanes'] if x['gpu']==args.gpu)
    folder=manifest.parent/('gpu%d'%args.gpu);folder.mkdir(exist_ok=True)
    lock=open(folder/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=folder/'status.json'
    origin=read(state_path)['started_unix'] if state_path.exists() else entered
    train_end,end=deadlines(dict(plan,started_unix=origin),lane)
    gate=ReviewGate(manifest,args.approval,lane['arm']);gate.verify_inputs();gate()
    if state_path.exists():
        if not args.resume:raise RuntimeError('Existing lane requires explicit --resume')
        state=read(state_path)
        if state['status'] in ('stopped','completed'):raise RuntimeError('Refusing to resume a stopped/completed lane')
        assert state['manifest_sha256']==file_digest(manifest)
        assert state['deadline_unix']==end and state['training_deadline_unix']==train_end
        assert file_digest(state['checkpoint'])==state['checkpoint_sha256']
        assert file_digest(state['cache'])==state['cache_sha256']
        write_json(folder/('before-resume-%d.json'%time.time()),state)
    else:
        assert not args.resume
        state=dict(status='running',cycle=1,checkpoint=plan['checkpoint'],cache=plan['cache'],
            checkpoint_sha256=file_digest(plan['checkpoint']),cache_sha256=file_digest(plan['cache']),
            manifest_sha256=file_digest(manifest),started_unix=origin,
            training_deadline_unix=train_end,deadline_unix=end,
            next_dev_index=0,physical_frames=0,sm_updates=0,recent=[],regressions=0)
        write_json(folder/'best-policy.json',dict(checkpoint=plan['checkpoint'],evaluation=plan['baseline']))
    state.update(pid=os.getpid(),status='running',gpu=args.gpu,arm=lane['arm']);child=None
    def persist(**kw):
        state.update(**kw,updated_unix=time.time());write_json(state_path,state)
    def stop_child():
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    def stopped(signum,frame):
        stop_child();persist(status='stopped',signal=signum);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    persist()
    events=open(folder/'events.jsonl','a',buffering=1)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(folder/'tensorboard'))
    def emit(kind,row):
        event=dict(kind=kind,cycle=state['cycle'],unix=time.time(),**row)
        events.write(json.dumps(event,allow_nan=False)+'\n')
        for k,v in row.items():
            if isinstance(v,(int,float)):writer.add_scalar(kind+'/'+k,v,state['cycle'])
        writer.flush()
    def execute(name,command,artifacts,until,validation=None,allow_prediction_rejection=False):
        nonlocal child
        stage=folder/('cycle-%04d'%state['cycle']);stage.mkdir(exist_ok=True)
        receipt=stage/(name+'.receipt.json')
        command=[sys.executable]+[str(x) for x in command]
        if receipt.exists():return verify_receipt(receipt,command)
        logfile=stage/(name+'.log')
        if logfile.exists():raise RuntimeError('Incomplete stage preserved; diagnose before resuming: '+str(logfile))
        if time.time()>=until:raise TimeoutError('Fixed phase deadline reached')
        gate.verify_inputs();gate();started=time.time()
        if started>=until:raise TimeoutError('Fixed phase deadline reached during input verification')
        with open(logfile,'x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            persist(phase=name,child_pid=child.pid,log=str(logfile))
            while child.poll() is None:
                if time.time()>=until:
                    stop_child();raise TimeoutError('Fixed phase deadline reached')
                persist();time.sleep(5)
        if child.returncode:
            if expected_worker_deadline(child.returncode,logfile.read_text(),time.time(),until):
                raise TimeoutError('Worker stopped at the fixed deadline; incomplete stage preserved')
            rejected=(allow_prediction_rejection and child.returncode==1 and Path(artifacts[0]).is_file() and
                      read(artifacts[0]).get('passed') is False and
                      logfile.read_text().rstrip().endswith('AssertionError: Prediction validation failed'))
            if not rejected:raise RuntimeError('%s exited %s; %s'%(name,child.returncode,logfile))
        proof=validation() if validation else {}
        result=dict(command=command,proof=proof,seconds=time.time()-started,
                    artifacts={str(p):file_digest(p) for p in artifacts})
        # Use the worker's cumulative update counter, including across resumes.
        metric_path=Path(command[command.index('--output')+1])/'metrics.jsonl' if '--output' in command else None
        if metric_path is not None and metric_path.is_file():
            for line in metric_path.read_text().splitlines():
                row=json.loads(line);step=row.get('step',row.get('update',row.get('physical_frames',state['cycle'])))
                for key,value in row.items():
                    if isinstance(value,(int,float)):
                        if not math.isfinite(value):raise FloatingPointError('Nonfinite worker metric: '+key)
                        writer.add_scalar(name+'/'+key,value,step)
            writer.flush()
        write_json(receipt,result);emit('runtime',{name+'_seconds':result['seconds']})
        state.pop('child_pid',None);child=None
        return result
    def evaluate(checkpoint,seeds,name,until):
        out=folder/name;out.mkdir(exist_ok=True);rows=[]
        sha=file_digest(checkpoint)
        for seed in seeds:
            output=out/('%d.json'%seed)
            def check():
                r=read(output)
                assert r['checkpoint_sha256']==sha and r['weights_unchanged']
                assert r['seed']==seed and r['episodes']==len(r['returns'])==8
                assert all(math.isfinite(x) for x in r['returns'])
                return dict(checkpoint_sha256=sha,seed=seed)
            execute(name+'-'+str(seed),['scripts/diagnose_reward.py','--checkpoint',checkpoint,
                '--output',output,'--gpu',args.gpu,'--seed',seed,'--save-starts'],
                [output,output.with_suffix('.starts.pt')],until,check)
            rows.append(read(output))
        result=aggregate(rows);write_json(out/'summary.json',result);return result
    cli=['-m','unionmimic.cli']
    try:
        while time.time()<train_end:
            if shutil.disk_usage(folder).free<50*1024**3:raise RuntimeError('Less than 50 GiB free')
            cycle=state['cycle'];out=folder/('cycle-%04d'%cycle);out.mkdir(exist_ok=True)
            checkpoint=Path(state['checkpoint']);cache=Path(state['cache'])
            prediction=None
            if lane['arm']=='dynamics':
                # Use the unchanged teacher for collection so the online
                # representation cannot silently alter the exploration task.
                data=out/'data'
                command=['collect','--checkpoint',plan['checkpoint'],'--output',data,'--gpu',args.gpu,
                    '--envs',16,'--steps',600,'--probe-noise',plan['dynamics']['probe_noise'],
                    '--stress-every',2,'--seed',lane['seed']+1000+cycle,'--gravity-max-degrees',1.0]
                execute('collect',cli+command,[data/'episodes.h5',data/'collection.json'],train_end,
                        lambda:validate_completed_stage('collect',command))
                recent=(state['recent']+[str(data)])[-3:];dataset=out/'dataset'
                command=['merge','--data',plan['dataset'],*recent,'--output',dataset]
                execute('merge',cli+command,[dataset/'episodes.h5',dataset/'collection.json'],train_end,
                        lambda:validate_completed_stage('merge',command))
                target=out/'dynamics'
                def check_dynamics():
                    r=read(target/'completed.json')
                    assert all(r[k] for k in ('actor_unchanged','encoder_unchanged','image_decoder_unchanged','dynamics_changed'))
                    assert r['step']>r['restored_step']
                    return r
                execute('dynamics',['scripts/train_dynamics_segment.py','--data',dataset,'--checkpoint',checkpoint,
                    '--output',target,'--gpu',args.gpu,'--seed',lane['seed'],'--updates',250,'--deadline',train_end,
                    '--manifest',manifest,'--approval',args.approval],
                    [target/'last.pt',target/'completed.json'],train_end,check_dynamics)
                current=target/'last.pt';updates=read(target/'completed.json')['step']
                prepared=out/'prepared'
                # Keep selection on the same original episode holdout. The
                # rolling collection split must not redefine the benchmark.
                command=['prepare','--data',plan['dataset'],'--checkpoint',current,'--output',prepared,'--gpu',args.gpu,'--recompute-state']
                execute('prepare',cli+command,[prepared/'cache.pt',prepared/'recovery.json'],train_end,
                        lambda:validate_completed_stage('prepare',command))
                cache=prepared/'cache.pt';prediction_file=out/'prediction.json'
                def check_prediction():
                    r=read(prediction_file)
                    assert r['checkpoint_sha256']==file_digest(current)
                    assert r['cache_metadata']['state_model_sha256']==read(prepared/'recovery.json')['state_model_sha256']
                    assert set(r['horizons'])=={'1','4'} and isinstance(r['passed'],bool)
                    return dict(passed=r['passed'])
                execute('prediction',cli+['validate','--checkpoint',current,'--cache',cache,'--output',prediction_file,'--gpu',args.gpu],
                    [prediction_file],train_end,check_prediction,allow_prediction_rejection=True)
                prediction=read(prediction_file)
                emit('prediction',dict(passed=prediction['passed'],**{f'h{h}_{k}':v[k] for h,v in prediction['horizons'].items() for k in ('reward_rmse','joint_rmse')}))
                frames=state['physical_frames']+read(data/'collection.json')['transitions'];state['recent']=recent
            elif lane['arm']=='ppo':
                target=out/'ppo'
                command=['scripts/train_ppo_segment.py','--checkpoint',checkpoint,'--output',target,
                    '--gpu',args.gpu,'--seed',lane['seed']+cycle,'--epochs',16,'--deadline',train_end,
                    '--manifest',manifest,'--approval',args.approval]
                def check_ppo():
                    r=read(target/'completed.json')
                    assert r['frozen_modules_unchanged'] and r['actor_changed']
                    assert r['epoch']>r['restored_epoch'] and r['actor_steps']>0
                    return r
                execute('ppo',command,[target/'last.pt',target/'completed.json'],train_end,check_ppo)
                current=target/'last.pt';frames=read(target/'completed.json')['physical_frames'];updates=0
            else:
                target=out/'sm'
                command=['sm','--cache',cache,'--checkpoint',checkpoint,'--output',target,'--gpu',args.gpu,
                    '--seed',lane['seed'],'--updates',500,'--lr',plan['sm']['actor_lr'],
                    '--anchor-coef',plan['sm']['anchor_coefficient']]+(['--resume'] if cycle>1 else [])
                execute('sm',cli+command,[target/'last.pt',target/'completed.json'],train_end,
                    lambda:validate_completed_stage('sm',command))
                current=target/'last.pt';updates=read(target/'completed.json')['step'];frames=state['physical_frames']
            index=state['next_dev_index'];due=plan['development_hours']
            if index<len(due) and time.time()>=state['started_unix']+due[index]*3600:
                result=evaluate(current,plan['periodic_dev_seeds'],'dev-%03d'%cycle,train_end)
                accepted=stable(result,plan['baseline']) and (prediction is None or prediction['passed'])
                emit('development',{**result,'stable':accepted})
                state['regressions']=0 if accepted else state['regressions']+1
                best=read(folder/'best-policy.json')
                if prediction is None:
                    improves=(result['mean_return']>best['evaluation']['mean_return'] and
                              result['rotation_radians']>=plan['baseline']['rotation_radians'])
                    model_score=None
                else:
                    model_score=sum(v['reward_rmse']/v['reward_mean_baseline_rmse']+
                        v['joint_rmse']/v['joint_persistence_rmse']+v['failure']['false_positive_rate']
                        for v in prediction['horizons'].values())
                    improves=model_score<best.get('prediction_score',plan['baseline_prediction_score'])
                if accepted and improves:
                    shutil.copyfile(current,folder/'best-policy.pt.tmp');os.replace(folder/'best-policy.pt.tmp',folder/'best-policy.pt')
                    write_json(folder/'best-policy.json',dict(checkpoint=str(folder/'best-policy.pt'),evaluation=result,
                        prediction_score=model_score,prediction=prediction))
                state['next_dev_index']=index+1
                if state['regressions']>=2:raise RuntimeError('Repeated physical regression beyond the unchanged stability gate')
            if lane['arm']=='sm':
                data=out/'data'
                command=['collect','--checkpoint',current,'--output',data,'--gpu',args.gpu,
                         '--envs',16,'--steps',600,'--probe-noise',.25,'--seed',lane['seed']+1000+cycle,
                         '--gravity-max-degrees',1.0]
                execute('collect',cli+command,[data/'episodes.h5',data/'collection.json'],train_end,
                        lambda:validate_completed_stage('collect',command))
                recent=(state['recent']+[str(data)])[-3:];dataset=out/'dataset'
                command=['merge','--data',plan['dataset'],*recent,'--output',dataset]
                execute('merge',cli+command,[dataset/'episodes.h5',dataset/'collection.json'],train_end,
                        lambda:validate_completed_stage('merge',command))
                prepared=out/'prepared';command=['prepare','--data',dataset,'--checkpoint',current,
                        '--output',prepared,'--gpu',args.gpu]
                execute('prepare',cli+command,[prepared/'cache.pt',prepared/'recovery.json'],train_end,
                        lambda:validate_completed_stage('prepare',command))
                cache=prepared/'cache.pt';target=out/'calibrated'
                command=['calibrate','--cache',cache,'--checkpoint',current,'--output',target,
                         '--gpu',args.gpu,'--updates',1000,'--resume']
                execute('calibrate',cli+command,[target/'best.pt',target/'validation.json',target/'completed.json'],train_end,
                        lambda:validate_completed_stage('calibrate',command))
                if not read(target/'validation.json')['passed']:raise RuntimeError('Prediction gate failed; actor continuation prohibited')
                current=target/'best.pt';frames+=read(data/'collection.json')['transitions'];state['recent']=recent
            persist(checkpoint=str(current),checkpoint_sha256=file_digest(current),cache=str(cache),
                cache_sha256=file_digest(cache),physical_frames=frames,
                sm_updates=updates if lane['arm']=='sm' else 0,dynamics_updates=updates if lane['arm']=='dynamics' else 0,
                cycle=cycle+1,phase='committed')
            emit('progress',dict(physical_frames=frames,sm_updates=state['sm_updates'],
                dynamics_updates=state.get('dynamics_updates',0),training_hours=(time.time()-state['started_unix'])/3600))
            best=read(folder/'best-policy.json')
            prune_bulk_outputs(folder,cycle,3,[current,cache,Path(best['checkpoint'])]+[Path(p)/'episodes.h5' for p in state['recent']])
    except TimeoutError:
        if time.time()<train_end:raise
        persist(phase='training_budget_reached')
    except BaseException as exc:
        stop_child()
        if state['status']!='stopped':persist(status='failed',error=repr(exc),traceback=traceback.format_exc())
        writer.close();events.close();raise
    try:
        best=read(folder/'best-policy.json');persist(phase='final_development',selected_checkpoint=best['checkpoint'])
        # Selection uses development data only. Once selected, the locked set
        # is evaluated exactly once and is never used to change the policy.
        development=evaluate(best['checkpoint'],plan['full_dev_seeds'],'final-dev',end)
        persist(phase='locked_test',locked_test_opened=True)
        test=evaluate(best['checkpoint'],plan['test_seeds'],'final-test',end)
        write_json(folder/'final-result.json',dict(selected=best,development=development,test=test,
            seed=lane['seed'],arm=lane['arm'],source_manifest_sha256=file_digest(manifest)))
        if args.gpu==0:
            reference=(test if test['checkpoint_sha256']==file_digest(plan['checkpoint']) else
                       evaluate(plan['checkpoint'],plan['test_seeds'],'baseline-test',end))
            write_json(manifest.parent/'baseline-test.json',reference)
        persist(status='completed',phase='finished');emit('completion',dict(seconds=time.time()-state['started_unix']))
    except BaseException as exc:
        stop_child()
        if state['status']!='stopped':persist(status='evaluation_incomplete',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:writer.close();events.close()
