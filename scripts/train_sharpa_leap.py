#!/usr/bin/env python3
"""Sharpa runner around generated pinned-reference updates, with a one-shot gate."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym  # noqa: F401 -- must precede torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa'))
import argparse,copy,hashlib,json,random,signal,time,traceback,subprocess,shutil
from types import SimpleNamespace
import numpy as np
import torch
from in_hand_rotation_mjlab.world_model.policy import Actor,Critic
from native_update import fit_update,actor_critic_update
from unionmimic.common import write_json,file_digest,digest,save_checkpoint
from unionmimic.leap_sharpa.model import SharpaModel,PhysicalError,SharpaTask,DC
from unionmimic.leap_sharpa.simulator import make_simulator,SharpaCollector,SharpaReplay


def finite(modules):
    return all(torch.isfinite(v).all() for m in modules for v in m.state_dict().values() if torch.is_tensor(v))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--hours',type=float,default=.75);ap.add_argument('--seed',type=int,default=20261104)
    ap.add_argument('--engineering-check',action='store_true');ap.add_argument('--continue-after-gate',action='store_true')
    ap.add_argument('--operator-evidence')
    ap.add_argument('--authorization',help='Explicit scoped authorization JSON for a new run')
    ap.add_argument('--quick-plan',help='Separately authorized one-hour comparison plan')
    ap.add_argument('--spin-sign',type=int,choices=(-1,1),default=1)
    cli=ap.parse_args();run=Path(cli.run).resolve()
    if run.exists():raise FileExistsError(run)
    if not 0<cli.hours<=12:raise ValueError('Hours must be in (0,12]')
    if cli.hours>1 and not cli.continue_after_gate:raise ValueError('Long execution needs the authorized gate')
    if cli.engineering_check and cli.continue_after_gate:raise ValueError('Operator diagnostic cannot launch training')
    quick=None
    if cli.quick_plan:
        quick=json.loads(Path(cli.quick_plan).resolve().read_text())
        if cli.continue_after_gate or cli.engineering_check or cli.authorization:
            raise ValueError('Quick comparison requires its own plan and budget')
        assert quick['approved_by_user'] is True and quick['scope']=='sharpa_action_quick_sm'
        assert cli.hours==quick['hours']==1 and time.time()+3600<=quick['authorization_deadline_unix']
        assert (ROOT/quick['run']).resolve()==run and quick['gpu']==cli.gpu and quick['seed']==cli.seed
        assert quick['spin_sign']==cli.spin_sign and quick['model_warmup_rounds']==60
        assert quick['evaluation_elapsed_minutes']==[15,30] and quick['automatic_resume'] is False
        assert cli.operator_evidence and Path(cli.operator_evidence).resolve()==(ROOT/quick['pilot']/'operator-acceptance.json').resolve()
    if cli.continue_after_gate or quick:
        if not cli.operator_evidence:raise ValueError('Verified operator evidence is required')
        ev=Path(cli.operator_evidence).resolve();accepted=json.loads(ev.read_text())
        assert accepted['passed'] and accepted['model_updates']==10 and accepted['actor_updates']==1 and accepted['critic_optimizer_steps']==20
        old_manifest=json.loads((ev.parent/'manifest.json').read_text())
        assert all(file_digest(ROOT/p)==h for p,h in old_manifest['sources'].items()),'Operator sources have changed'
        pilot_eval=json.loads((ev.parent/'evaluation.json').read_text())
        assert pilot_eval['passed'] and pilot_eval.get('spin_sign',1)==cli.spin_sign,'Independent physical evaluation has not passed'
        if quick:assert pilot_eval['checkpoint_sha256']==file_digest(ev.parent/'ckpt/operator.pt')
    authorization=None
    if cli.authorization:
        authorization=json.loads(Path(cli.authorization).resolve().read_text())
        assert authorization.get('user_authorized') is True
        assert authorization.get('scope')=='new_sharpa_sm_baseline_10hz'
        assert time.time()+cli.hours*3600<=authorization['deadline_unix'],'Run exceeds authorized window'
        assert cli.hours<=authorization['max_run_hours'],'Run exceeds scoped duration'
    run.mkdir(parents=True);(run/'ckpt').mkdir();started=time.time();deadline=started+cli.hours*3600
    args=SimpleNamespace(envs=4096,replay=1000,wm_batch=2048,wm_steps=10,wm_k=8,starts=4096,horizon=16,
        wm_only=quick['model_warmup_rounds'] if quick else 300,utd=.4,warmup=64,lr=2e-4,gamma=.99,lam=.95,imag_noise=0.,imag_noise_final=None,imag_noise_updates=6000)
    paths=[Path(__file__).resolve(),ROOT/'scripts/prepare_leap_sharpa.py',ROOT/'scripts/evaluate_sharpa_leap.py',ROOT/'unionmimic/sharpa_leap.py',ROOT/'scripts/env.sh',ROOT/'config/runtime-lock.json']
    paths+=list((ROOT/'unionmimic/leap_sharpa').glob('*.py'))
    paths+=list((ROOT/'work/runtime/leap-sharpa').rglob('*.py'))
    paths+=[ROOT/'unionmimic/online_baseline'/n for n in ('simulator.py','task.py','core.py','curriculum.py')]
    paths+=[ROOT/'work/upstreams/WM-Craftnet/isaacgymenvs/tasks'/n for n in ('wm_craftnet_rotation.py','wm_craftnet_camera_rotation.py','base/vec_task.py')]
    paths+=[ROOT/'work/upstreams/WM-Craftnet/isaacgymenvs/cfg'/n for n in ('config.yaml','task/WMCraftnetRotation.yaml','train/WMCraftnetPPO.yaml')]
    sources={str(p.relative_to(ROOT)):file_digest(p) for p in paths}
    write_json(run/'manifest.json',dict(started_unix=started,deadline_unix=deadline,cli=vars(cli),config=vars(args),sources=sources,
        upstream_commit='0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0',gate_updates=350,
        approval=quick or authorization or 'User explicitly authorized training after complete code/short-run verification, 2026-11-03 23:24',
        long_training_approved=cli.continue_after_gate,from_scratch=True,prior_loaded=False,auto_resume=False))
    for p in paths:
        target=run/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    stage='initializing';updates=0;collector=None
    def status(value,**kw):
        write_json(run/'status.json',dict(status=value,phase=stage,pid=os.getpid(),gpu=cli.gpu,updated_unix=time.time(),
            started_unix=started,deadline_unix=deadline,updates=updates,actor_updates=max(0,updates-args.wm_only),**kw))
    def stop(*_):raise InterruptedError('Stop requested; no automatic resume')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    event=(run/'events.jsonl').open('a',buffering=1)
    def emit(kind,**kw):
        row=dict(kind=kind,unix=time.time(),**kw);event.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row,allow_nan=False),flush=True)
    try:
        status('running');torch.set_num_threads(4)
        env,native,cfg=make_simulator(4096,cli.gpu,cli.seed,spin_sign=cli.spin_sign);dev=env.device
        assert bool((env.spin_axis[:,2]==cli.spin_sign).all())
        assert env.num_actions==22 and env.clip_actions==1 and list(env.actuated_dof_indices.cpu().numpy())==list(range(22))
        assert len(env.object_rb_handles)==1,'External-force adapter is for the current single-body object'
        native.cfg.update(force_scale=float(env.force_scale),force_probability=float(env.random_force_prob_scalar),force_decay=float(env.force_decay))
        write_json(run/'native-config.json',cfg);write_json(run/'task-constants.json',native.cfg)
        if quick:torch.manual_seed(cli.seed+1)
        actor=Actor(730,22).to(dev)
        critic=Critic(279).to(dev) if quick else None
        wm=SharpaModel(native.cfg).to(dev);err=PhysicalError().to(dev)
        task=SharpaTask(wm,native.cfg,dev);replay=SharpaReplay(4096,dev)
        critic_dim=task.critic_features(torch.zeros(2,177,device=dev),None,None,torch.zeros(2,100,device=dev)).shape[-1]
        if quick:assert critic_dim==279
        else:critic=Critic(critic_dim).to(dev)
        opt_wm=torch.optim.Adam(wm.parameters(),lr=args.lr)
        opt_pi=torch.optim.Adam(actor.parameters(),lr=args.lr,betas=(.9,.95));opt_v=torch.optim.Adam(critic.parameters(),lr=args.lr)
        initial=dict(actor=digest(actor.net),critic=digest(critic.net),model=digest(wm))
        collector=SharpaCollector(env,native,actor,replay)
        stage='initial_collection';status('running')
        for _ in range(64):collector.collect(.5)
        mask=replay.epi[:64]>=0
        wm.fit_stats(replay.s[:64][mask],replay.s_next[:64][mask],replay.ctx[:64][mask])
        err.fit(replay.s[:64][mask],replay.s_next[:64][mask]);dt=native.cfg['control_dt'];rmax=task.max_step_reward()
        del mask
        emit('initialized',seconds=time.time()-started,config=vars(args),valid_rows=collector.valid_rows,r_max=rmax,control_dt=dt,
             initial_actor=initial['actor'],initial_critic=initial['critic'])
        status('running');last_save=time.time();last_eval=time.time();history=[];passed=False;acc=0.;best_return=-float('inf')
        def save(name):
            save_checkpoint(run/'ckpt'/name,dict(actor=actor.checkpoint(),wm=wm.state_dict(),critic=critic.state_dict(),error=err.state_dict(),
                optimizers=[o.state_dict() for o in (opt_wm,opt_pi,opt_v)],updates=updates,task_constants=native.cfg,
                rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),
                replay_included=False,exact_simulator_resume=False,initial=initial,spin_sign=cli.spin_sign))
        def evaluate(name,checkpoint):
            nonlocal last_eval,best_return
            checkpoint_elapsed=checkpoint.stat().st_mtime-started
            output=run/(name+'.json')
            remaining=deadline-time.time()
            if remaining<30:raise TimeoutError('Insufficient original budget for required evaluation')
            with (run/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_sharpa_leap.py'),'--gpu',str(cli.gpu),
                    '--checkpoint',str(checkpoint),'--out',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(600,remaining),check=True)
            result=json.loads(output.read_text());assert result['passed'] and result['episodes']==512
            emit('evaluation',name=name,update=updates,checkpoint_elapsed_seconds=checkpoint_elapsed,
                 valid_training_transitions=collector.valid_rows,**{k:v for k,v in result.items() if k!='rows'})
            if result['mean_return']>best_return:
                best_return=result['mean_return'];shutil.copyfile(checkpoint,run/'ckpt/best.pt')
                write_json(run/'best-evaluation.json',dict(evaluation=name,mean_return=best_return,selection='maximum native mean return on fixed development seed'))
            last_eval=time.time();return result
        if cli.engineering_check:
            stage='operator_check'
            fm=fit_update(wm,err,replay,opt_wm,args,dev,4096,wm.normalize)
            old_model=digest(wm);old_actor=digest(actor.net);old_critic=digest(critic.net)
            am=actor_critic_update(actor,critic,wm,task,replay,opt_pi,opt_v,args,301,dev,wm.normalize,dt,rmax)
            assert old_model==digest(wm),'Actor/critic update mutated frozen world model'
            assert old_actor!=digest(actor.net) and old_critic!=digest(critic.net)
            assert finite([actor,critic,wm,err]) and am['finite'] and am['gn']>0
            collector.actor_updates=1
            for _ in range(8):collector.collect(.1)
            result=dict(passed=True,full_recipe_short_training=False,operator_check_only=True,model_updates=10,actor_updates=1,
                        critic_optimizer_steps=20,parity=collector.max_error,used_actor_updates=collector.used_actor_updates,
                        frozen_model_unchanged=True,model=fm,actor=am,seconds=time.time()-started)
            save('operator.pt');write_json(run/'operator-acceptance.json',result);emit('operator_passed',**result);status('completed');return
        stage='quick_training' if quick else 'pilot';status('running')
        training_deadline=deadline-600 if cli.continue_after_gate or quick else deadline
        eval_schedule=list(quick['evaluation_elapsed_minutes']) if quick else []
        timings=dict(collect=0.,fit=0.,actor_critic=0.)
        while time.time()<training_deadline:
            phase_start=time.perf_counter()
            collector.collect(.1);acc+=args.utd
            timings['collect']+=time.perf_counter()-phase_start
            if acc<1:continue
            acc-=1;updates+=1
            phase_start=time.perf_counter()
            fm=fit_update(wm,err,replay,opt_wm,args,dev,4096,wm.normalize);am={}
            timings['fit']+=time.perf_counter()-phase_start
            if updates>args.wm_only:
                phase_start=time.perf_counter()
                am=actor_critic_update(actor,critic,wm,task,replay,opt_pi,opt_v,args,updates,dev,wm.normalize,dt,rmax)
                timings['actor_critic']+=time.perf_counter()-phase_start
                collector.actor_updates=updates-args.wm_only
            history.append(dict(update=updates,**fm,**am))
            if updates%10==0:
                assert finite([actor,critic,wm,err]),'Nonfinite saved model'
                status('running',model_updates=updates*10,critic_optimizer_steps=max(0,updates-args.wm_only)*20,valid_rows=collector.valid_rows)
                rows=torch.arange(max(0,replay.G-25),replay.G,device=dev)%replay.L
                valid=replay.epi[rows]>=0;real=replay.rew[rows][valid].mean(0)
                emit('update',update=updates,frames=replay.G*4096,seconds=time.time()-started,
                    real_reward_per_s=float(real[0]/dt),real_rotation_rad_per_s=float(real[1]/dt),
                    cumulative_failures=collector.ep_drops,cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    valid_training_transitions=collector.valid_rows,phase_wall_seconds=dict(timings),**fm,**am)
            if quick:
                if time.time()-last_save>=300 and updates>args.wm_only:
                    save('latest.pt');last_save=time.time()
                if eval_schedule and time.time()-started>=eval_schedule[0]*60 and deadline-time.time()>1200:
                    name='evaluation-%06d'%updates
                    checkpoint=run/'ckpt'/('actor-%06d.pt'%updates)
                    save(checkpoint.name);evaluate(name,checkpoint)
                    while eval_schedule and time.time()-started>=eval_schedule[0]*60:eval_schedule.pop(0)
                continue
            request=run/'promote-request.json'
            early=False
            if request.exists():
                promotion=json.loads(request.read_text())
                early=promotion.get('action')=='promote' and promotion.get('explicit_user_request') is True
            threshold=302 if early else 350
            if updates>=threshold and not passed:
                assert collector.used_actor_updates>=(1 if early else 20) and all(h.get('finite',True) for h in history)
                assert finite([actor,critic,wm,err])
                assert all(file_digest(ROOT/p)==h for p,h in sources.items()),'Source changed during pilot'
                final=dict(actor=digest(actor.net),critic=digest(critic.net),model=digest(wm))
                assert all(final[k]!=initial[k] for k in initial),'Unchanged trainable module'
                assert max(h.get('gn',0) for h in history)>0
                result=dict(passed=True,full_recipe_short_training=not early,abbreviated_by_explicit_user_request=early,learning_gain_verified=False,
                    updates=updates,model_updates=updates*10,actor_updates=updates-300,critic_optimizer_steps=(updates-300)*20,
                    used_actor_updates=collector.used_actor_updates,parity=collector.max_error,initial=initial,final=final,
                    seconds=time.time()-started,source_hashes_unchanged=True,actual_config=vars(args))
                save('pilot-verified.pt');stage='pilot_evaluation';status('running')
                result['evaluation']=evaluate('evaluation-pilot',run/'ckpt/pilot-verified.pt')
                write_json(run/'pilot-acceptance.json',result);emit('pilot_passed',**{k:v for k,v in result.items() if k!='evaluation'});passed=True
                if not cli.continue_after_gate:break
                stage='long_training';status('running');emit('long_training_started',deadline_unix=deadline)
            if time.time()-last_save>=300 and updates>300:
                save('latest.pt');save('checkpoint-cycle-%06d.pt'%updates);last_save=time.time()
            if passed and time.time()-last_eval>=7200 and deadline-time.time()>1200:
                save('latest.pt');evaluate('evaluation-%06d'%updates,run/'ckpt/latest.pt')
            if stage=='pilot' and time.time()-started>3600:raise TimeoutError('Full pilot did not pass within one hour')
        save('final.pt')
        if quick:
            assert collector.used_actor_updates>=1 and updates>args.wm_only,'No completed actor update and recollection'
            assert all(file_digest(ROOT/p)==h for p,h in sources.items()),'Sources changed during quick training'
            evaluate('evaluation-final',run/'ckpt/final.pt')
            stage='finished';status('completed');emit('finished',updates=updates,phase_wall_seconds=timings);return
        if not passed:raise RuntimeError('Budget ended before full pilot acceptance; long training not started')
        if cli.continue_after_gate:evaluate('evaluation-final',run/'ckpt/final.pt')
        stage='finished';status('completed');emit('finished',updates=updates)
    except BaseException as exc:
        status('stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc));raise


if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
