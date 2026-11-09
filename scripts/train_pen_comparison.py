"""Authorized deadline-bounded training using the same adapters as the operator checks."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'));sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa'))
import argparse,json,time,signal,subprocess,traceback,shutil
from types import SimpleNamespace
import torch
from in_hand_rotation_mjlab.world_model.policy import Actor,Critic
from native_update import fit_update,actor_critic_update
from unionmimic.common import write_json,file_digest,save_checkpoint
from unionmimic.training_window import remaining_seconds
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.leap_sharpa.pen_model import PenModel,PenTask,constants
from unionmimic.leap_sharpa.pen_collect import PenCollector,PenReplay,snapshot
from unionmimic.leap_sharpa.model import PhysicalError
from unionmimic.leap_sharpa.paired_ppo import PairedPPO,RECIPE

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--algorithm',choices=['sm','ppo'],required=True)
    p.add_argument('--task-set',choices=['all','reference'],required=True);p.add_argument('--retarget',required=True)
    p.add_argument('--run',required=True);p.add_argument('--authorization',required=True);p.add_argument('--gate',required=True)
    a=p.parse_args();auth=json.loads(Path(a.authorization).read_text());remaining_seconds(auth)
    gate=json.loads(Path(a.gate).read_text());root=Path(a.retarget).resolve()
    assert gate['passed'] and gate['retarget_manifest_sha256']==file_digest(root/'manifest.json')
    run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);start=time.time();deadline=auth['deadline_unix'];torch.set_num_threads(4)
    args=SimpleNamespace(envs=4096,replay=1000,wm_batch=2048,wm_steps=10,wm_k=8,starts=4096,horizon=16,wm_only=300,
        utd=.4,warmup=64,lr=2e-4,gamma=.99,lam=.95,imag_noise=0.,imag_noise_final=None,imag_noise_updates=6000)
    paths=[Path(__file__).resolve(),ROOT/'scripts/evaluate_pen_policy.py',ROOT/'assets/reference_pen/pen_tracking220.urdf']
    paths+=list((ROOT/'unionmimic').glob('pen*.py'))+list((ROOT/'unionmimic/leap_sharpa').glob('*.py'))+list((ROOT/'work/runtime/leap-sharpa').rglob('*.py'))
    paths+=[ROOT/'unionmimic'/x for x in ['hand_kinematics.py','motions.py','sharpa_leap.py','training_window.py','online_baseline/simulator.py','online_baseline/task.py','online_baseline/core.py']]
    paths+=[ROOT/'work/upstreams/WM-Craftnet/isaacgymenvs/tasks'/x for x in ['wm_craftnet_rotation.py','wm_craftnet_camera_rotation.py','base/vec_task.py']]
    sources={str(f.relative_to(ROOT)):file_digest(f) for f in paths}
    write_json(run/'manifest.json',dict(cli=vars(a),started_unix=start,deadline_unix=deadline,authorization=auth,sources=sources,
        sm_recipe=vars(args),ppo_recipe=RECIPE,physical_gate=gate,from_scratch=True,prior_loaded=False,wandb=False,
        known_deviations=['Sharpa 22 joints, pen residual controller and 50Hz','Sharpa native friction/PD randomization',
          'seven Wuji reward components; no inferred self-collision count from net contact forces',
          'retargeted joint/position references with immutable timestamps and object orientations'],auto_resume=False))
    for f in paths:
        dest=run/'source'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    events=(run/'events.jsonl').open('a',buffering=1);iteration=0;actor_updates=0;collector=None
    def emit(stage,**kw):
        row=dict(stage=stage,pid=os.getpid(),gpu=a.gpu,unix=time.time(),elapsed=time.time()-start,iteration=iteration,actor_updates=actor_updates);row.update(kw)
        write_json(run/'status.json',row);events.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
    def stop(*_):raise InterruptedError('Stopped; no automatic restart or budget extension')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        emit('initializing');env,cfg=make_pen_simulator(4096,a.gpu,20261190,root);dev=env.device
        if a.task_set=='reference':
            env.task_ids[:]=torch.arange(4096,device=dev)%2+2;env.reset_buf.fill_(1);env.step(torch.zeros(4096,22,device=dev))
        conf=constants(env);write_json(run/'environment.json',cfg);actor=Actor(1740,22).to(dev);model=PenModel(conf,env.pen_tasks).to(dev);task=PenTask(model,conf,dev)
        s,c=snapshot(env);critic=Critic(task.critic_features(s,None,None,model.normalize(c)).shape[-1]).to(dev)
        err=PhysicalError().to(dev)
        if a.algorithm=='sm':
            replay=PenReplay(4096,dev);collector=PenCollector(env,task,actor,replay=replay)
            ow=torch.optim.Adam(model.parameters(),lr=args.lr);op=torch.optim.Adam(actor.parameters(),lr=args.lr,betas=(.9,.95));ov=torch.optim.Adam(critic.parameters(),lr=args.lr)
            for _ in range(64):remaining_seconds(auth);collector.collect(.5)
            mask=replay.epi[:64]>=0;model.fit_stats(replay.s[:64][mask],replay.s_next[:64][mask],replay.ctx[:64][mask]);err.fit(replay.s[:64][mask],replay.s_next[:64][mask])
        else:
            learner=PairedPPO(actor,critic);collector=PenCollector(env,task,actor,critic=critic,ppo=learner);contexts=[]
            for _ in range(64):remaining_seconds(auth);collector.collect();contexts.append(snapshot(env)[1])
            c=torch.cat(contexts);model.c_mu.copy_(c.mean(0));model.c_sd.copy_(c.std(0).clamp_min(1e-6));del contexts,c
        def save(path):
            assert all(torch.isfinite(v).all() for m in (actor,critic,model) for v in m.state_dict().values()),'Nonfinite weights; preserve previous checkpoint'
            checkpoint=dict(actor=actor.checkpoint(),critic=critic.state_dict(),model=model.state_dict(),error=err.state_dict(),
                task_constants=conf,iteration=iteration,actor_updates=actor_updates,reference_sampler=env.reference_sampler.state_dict(),
                replay_included=False,exact_simulator_resume=False,retarget_manifest_sha256=file_digest(root/'manifest.json'))
            if a.algorithm=='sm':checkpoint['optimizers']=[o.state_dict() for o in (ow,op,ov)]
            else:checkpoint.update(optimizer=learner.optimizer.state_dict(),log_std=learner.log_std.detach(),context_mean=model.c_mu,context_std=model.c_sd)
            save_checkpoint(path,checkpoint)
        schedule=[15,30,60,120,240,480,720,960,1200,1380];eval_seconds=0.;best=-float('inf');last_save=time.time();acc=0.
        def evaluate(label):
            nonlocal eval_seconds,best
            checkpoint=run/f'checkpoint-{label}.pt';save(checkpoint);began=time.time();out=run/f'evaluation-{label}.json'
            with (run/f'evaluation-{label}.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_pen_policy.py'),'--gpu',str(a.gpu),'--checkpoint',str(checkpoint),
                    '--retarget',str(root),'--task-set',a.task_set,'--out',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(600,remaining_seconds(auth)),check=True)
            result=json.loads(out.read_text());assert result['passed'];eval_seconds+=time.time()-began
            emit('evaluation',label=label,checkpoint_elapsed=began-start,evaluation_seconds=eval_seconds,summary=result['summary'])
            if result['mean_return']>best:
                best=result['mean_return'];shutil.copy2(checkpoint,run/'best.pt');write_json(run/'best-evaluation.json',dict(evaluation=str(out),mean_return=best,selection='development return; all task precision/success metrics retained'))
        emit('warmup_collected',valid_rows=collector.valid_rows,parity=collector.max_error)
        while time.time()<deadline-600:
            if a.algorithm=='sm':
                collector.collect(.1);acc+=args.utd
                if acc<1:continue
                acc-=1;iteration+=1;metrics=fit_update(model,err,replay,ow,args,dev,4096,model.normalize)
                if iteration>300:
                    metrics.update(actor_critic_update(actor,critic,model,task,replay,op,ov,args,iteration,dev,model.normalize,.02,.24))
                    actor_updates+=1;collector.actor_updates=actor_updates
            else:
                metrics=learner.update([collector.collect() for _ in range(32)]);iteration+=1;actor_updates+=1;collector.actor_updates=actor_updates
            if iteration%10==0 or (a.algorithm=='sm' and iteration in (300,302)):
                emit('training' if actor_updates else 'model_warmup',valid_rows=collector.valid_rows,parity=collector.max_error,
                     evaluation_seconds=eval_seconds,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,metrics=metrics)
            if time.time()-last_save>=300:save(run/'latest.pt');last_save=time.time()
            if schedule and time.time()-start>=schedule[0]*60 and deadline-time.time()>900:
                evaluate('%06d'%iteration)
                while schedule and time.time()-start>=schedule[0]*60:schedule.pop(0)
        save(run/'final.pt')
        if deadline-time.time()>60:evaluate('final')
        assert all(file_digest(ROOT/f)==h for f,h in sources.items()),'Training source was modified'
        emit('completed',valid_rows=collector.valid_rows,evaluation_seconds=eval_seconds)
    except BaseException as exc:
        emit('stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc));raise
if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
