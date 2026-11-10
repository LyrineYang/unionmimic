"""Full-scale PPO pilot; a long run requires a separately approved concrete plan."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import argparse,json,time,traceback,subprocess,shutil,signal
import torch
from in_hand_rotation_mjlab.world_model.policy import Actor,Critic
from unionmimic.common import write_json,file_digest,digest,save_checkpoint
from unionmimic.online_baseline.simulator import make_simulator,state,context
from unionmimic.leap_sharpa.model import pack
from unionmimic.leap_sharpa.paired_ppo import PairedPPO,features,RECIPE
from unionmimic.ppo_plan import validate_plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--run',required=True)
    ap.add_argument('--iterations',type=int,default=5);ap.add_argument('--seed',type=int,default=20261104)
    ap.add_argument('--approved-plan')
    ap.add_argument('--spin-sign',type=int,choices=(-1,1),default=1)
    a=ap.parse_args();assert 2<=a.iterations<=20
    plan=None
    if a.approved_plan:
        plan=json.loads(Path(a.approved_plan).read_text())
        validate_plan(plan,RECIPE,a.run,a.seed,time.time(),ROOT)
        assert plan.get('gpu',a.gpu)==a.gpu,'GPU differs from the approved plan'
        assert plan.get('spin_sign',1)==a.spin_sign,'Rotation direction differs from the approved plan'
        pilot=ROOT/plan['pilot']
        assert json.loads((pilot/'acceptance.json').read_text())['passed']
        pilot_eval=json.loads((pilot/'evaluation.json').read_text())
        assert pilot_eval['passed'] and pilot_eval.get('spin_sign',1)==a.spin_sign
        if plan.get('verify_operator_sources'):
            pilot_manifest=json.loads((pilot/'manifest.json').read_text())
            assert all(file_digest(ROOT/p)==h for p,h in pilot_manifest['sources'].items()),'Operator sources have changed'
            assert pilot_eval['checkpoint_sha256']==file_digest(pilot/'actor.pt')
    run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(4)
    deadline=start+(plan['hours']*3600 if plan else 1200)
    sources=[Path(__file__),ROOT/'scripts/evaluate_sharpa_leap.py',ROOT/'unionmimic/ppo_plan.py',ROOT/'unionmimic/leap_sharpa/paired_ppo.py',ROOT/'unionmimic/leap_sharpa/model.py',ROOT/'unionmimic/online_baseline/simulator.py',
             ROOT/'work/upstreams/WM-Craftnet/rl_games/common/common_losses.py',
             ROOT/'work/upstreams/sm-reference-audit/src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_left_custom/rl_cfg.py']
    write_json(run/'manifest.json',dict(recipe=RECIPE,cli=vars(a),started_unix=start,from_scratch=True,ws_model_used=False,
        actor='730 -> 512/512/256 ELU tanh',critic='279 -> 256/128 LN SiLU',
        deviations_from_native_ppo=['SM-matched actor output/critic architecture and privileged features','validity mask for delayed Isaac resets','normalizers fixed during rollout/update'],
        sources={str(p.resolve().relative_to(ROOT)):file_digest(p) for p in sources},deadline_unix=deadline,approved_plan=plan))
    for p in sources:
        dest=run/'source'/p.resolve().relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    log=(run/'events.jsonl').open('a',buffering=1)
    def status(value,**kw):write_json(run/'status.json',dict(status=value,pid=os.getpid(),gpu=a.gpu,unix=time.time(),**kw))
    def stopped(*_):raise InterruptedError('User stop; no automatic resume')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        status('initializing');env,task,cfg=make_simulator(4096,a.gpu,a.seed,spin_sign=a.spin_sign);device=env.device
        assert bool((env.spin_axis[:,2]==a.spin_sign).all())
        if plan and plan.get('verify_operator_sources'):torch.manual_seed(a.seed+1)
        actor=Actor(730,22).to(device);critic=Critic(279).to(device);learner=PairedPPO(actor,critic)
        initial=dict(actor=digest(actor.net),critic=digest(critic.net))
        write_json(run/'initial.json',initial)
        history=env.obs_buf[:,:73,None].transpose(1,2).expand(-1,10,-1).clone()
        cm=torch.zeros(100,device=device);cs=torch.ones(100,device=device);errors=[];recollected=0
        def collect(warm=False):
            nonlocal history
            with torch.no_grad():
                invalid=env.reset_buf.bool().clone();valid=~invalid
                base=env.root_state_tensor[env.hand_indices,:13].clone();s=state(env);c=torch.cat((context(env),base[:,:7]),-1)
                ss=pack(s,base);x=features(ss,(c-cm)/cs);obs=history.flatten(1).clone()
                if warm and valid.any():actor.norm.update(obs[valid])
                dist=learner.distribution(obs);raw=dist.sample() if not warm else actor(obs)+.5*torch.randn(4096,22,device=device)
                val=critic(x);env.step(raw.clamp(-1,1));sn=state(env)
                r,f,t,_,m=task.reward(s,sn,c[:,:93]);nx=features(pack(sn,base),(c-cm)/cs)
                if valid.any():
                    errors.append(float((r[valid]-env.rew_buf[valid]).abs().max()))
                    assert errors[-1]<2e-3 and torch.equal((f|t)[valid],env.reset_buf.bool()[valid])
                frame=env.obs_buf[:,:73].clamp(-5,5);history=torch.cat((history[:,1:],frame[:,None]),1)
                history[invalid]=frame[invalid,None].expand(-1,10,-1)
                row=dict(history=obs,features=x,raw_action=raw,mean=dist.loc.clone(),std=dist.scale.clone(),logp=dist.log_prob(raw).sum(-1),
                    value=val,next_value=critic(nx),reward=env.rew_buf.clone(),failure=f,timeout=t,valid=valid)
                assert all(torch.isfinite(v).all() for v in row.values() if v.is_floating_point())
                return row,c
        warm=[]
        for _ in range(64):r,c=collect(True);warm.append(c[r['valid']])
        allc=torch.cat(warm);cm=allc.mean(0);cs=allc.std(0).clamp_min(1e-6);del allc,warm
        old_actor=digest(actor.net);old_critic=digest(critic.net)
        def save(path):
            save_checkpoint(path,dict(actor=actor.checkpoint(),critic=critic.state_dict(),optimizer=learner.optimizer.state_dict(),
                log_std=learner.log_std.detach(),context_mean=cm,context_std=cs,task_constants=task.cfg,iterations=i,
                exact_simulator_resume=False,spin_sign=a.spin_sign,initial=initial))
        best=-float('inf');last_eval=time.time();last_save=time.time();eval_seconds=0.;valid_total=0
        eval_schedule=list(plan.get('evaluation_elapsed_minutes',[120,240,360,480,600])) if plan else []
        def evaluate(label):
            nonlocal best,last_eval,eval_seconds
            checkpoint_seconds=time.time()-start;began=time.time()
            checkpoint=run/('actor-'+label+'.pt');save(checkpoint);output=run/('evaluation-'+label+'.json')
            with (run/('evaluation-'+label+'.log')).open('w') as stream:
                subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_sharpa_leap.py'),'--gpu',str(a.gpu),
                    '--checkpoint',str(checkpoint),'--out',str(output)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    timeout=min(600,deadline-time.time()),check=True)
            result=json.loads(output.read_text());assert result['passed'] and result['episodes']==512
            eval_seconds+=time.time()-began
            log.write(json.dumps(dict(kind='evaluation',label=label,iteration=i,checkpoint_elapsed_seconds=checkpoint_seconds,
                cumulative_evaluation_seconds=eval_seconds,environment_transitions=(64+32*i)*4096,valid_training_transitions=valid_total,
                **{k:v for k,v in result.items() if k!='rows'}))+'\n')
            if result['mean_return']>best:
                best=result['mean_return'];shutil.copyfile(checkpoint,run/'best.pt')
                write_json(run/'best-evaluation.json',dict(evaluation=label,mean_return=best))
            last_eval=time.time()
        i=0
        while i<a.iterations if not plan else time.time()<deadline-600:
            rows=[collect()[0] for _ in range(32)];metrics=learner.update(rows)
            valid_total+=metrics['valid_rows']
            if i:recollected+=1
            i+=1
            metrics.update(kind='update',iteration=i,elapsed=time.time()-start,evaluation_seconds=eval_seconds,
                environment_transitions=(64+32*i)*4096,valid_training_transitions=valid_total,
                reward_mean=float(torch.stack([x['reward'] for x in rows]).mean()))
            log.write(json.dumps(metrics)+'\n');print(json.dumps(metrics),flush=True);status('running',**metrics)
            if time.time()-last_save>=300:
                save(run/'latest.pt');save(run/('actor-cycle-%06d.pt'%i));last_save=time.time()
            if plan and eval_schedule and time.time()-start>=eval_schedule[0]*60 and deadline-time.time()>1200:
                evaluate('%06d'%i)
                while eval_schedule and eval_schedule[0]*60<=time.time()-start:eval_schedule.pop(0)
        assert digest(actor.net)!=old_actor and digest(critic.net)!=old_critic and recollected>=1
        assert all(torch.isfinite(p).all() for m in (actor,critic) for p in m.state_dict().values())
        save(run/'actor.pt')
        if plan:evaluate('final')
        write_json(run/'acceptance.json',dict(passed=True,learning_gain_verified=False,iterations=i,
            optimizer_steps=learner.steps,recollected_policy_iterations=recollected,reward_max_error=max(errors),
            seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
        status('completed')
    except BaseException as e:status('stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e));raise

if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
