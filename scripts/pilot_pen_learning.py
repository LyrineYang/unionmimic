"""Bounded full-scale operator check. No long-training or automatic promotion path."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'));sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa'))
import argparse,time,json,traceback,signal
from types import SimpleNamespace
import torch
from in_hand_rotation_mjlab.world_model.policy import Actor,Critic
from native_update import fit_update,actor_critic_update
from unionmimic.common import write_json,file_digest,digest,save_checkpoint
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.leap_sharpa.pen_model import PenModel,PenTask,constants,DX,DC,DF,H
from unionmimic.leap_sharpa.pen_collect import PenCollector,PenReplay,snapshot
from unionmimic.leap_sharpa.model import PhysicalError
from unionmimic.leap_sharpa.paired_ppo import PairedPPO,RECIPE


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--algorithm',choices=['sm','ppo'],required=True)
    ap.add_argument('--retarget',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=20261116)
    ap.add_argument('--full-warmup',action='store_true');ap.add_argument('--minutes',type=float,default=30)
    ap.add_argument('--task-set',choices=['all','reference'],default='all')
    a=ap.parse_args();assert 0<a.minutes<=60
    out=Path(a.out).resolve();a.retarget=str(Path(a.retarget).resolve());out.mkdir(parents=True,exist_ok=False);start=time.time();deadline=start+a.minutes*60
    torch.set_num_threads(4)
    args=SimpleNamespace(envs=4096,replay=1000,wm_batch=2048,wm_steps=10,wm_k=8,starts=4096,horizon=16,wm_only=300,
        utd=.4,warmup=64,lr=2e-4,gamma=.99,lam=.95,imag_noise=0.,imag_noise_final=None,imag_noise_updates=6000)
    paths=[Path(__file__).resolve(),ROOT/'assets/reference_pen/pen_tracking220.urdf']+list((ROOT/'unionmimic').glob('pen*.py'))
    paths+=list((ROOT/'unionmimic/leap_sharpa').glob('*.py'))+list((ROOT/'work/runtime/leap-sharpa').rglob('*.py'))
    paths+=[ROOT/'unionmimic/hand_kinematics.py',ROOT/'unionmimic/motions.py',ROOT/'unionmimic/online_baseline/simulator.py',ROOT/'unionmimic/online_baseline/task.py']
    sources={str(p.relative_to(ROOT)):file_digest(p) for p in paths}
    write_json(out/'manifest.json',dict(cli=vars(a),started_unix=start,deadline_unix=deadline,config=vars(args),ppo_recipe=RECIPE,
        sources=sources,retarget_manifest_sha256=file_digest(Path(a.retarget)/'manifest.json'),operator_only=not a.full_warmup,
        full_adaptation_accepted=False,known_deviations=['Sharpa collision configuration retained; Wuji self-collision pair penalty not available',
            'four-task allocation with source adaptive sampling conditioned on reference group','fixed composite 220mm pen and Sharpa native friction/PD randomization'],
        actor_input=DF*H,state_dim=DX,context_dim=DC,from_scratch=True,long_training_allowed=False))
    for p in paths:
        d=out/'source'/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(p.read_bytes())
    event=(out/'events.jsonl').open('a',buffering=1)
    def emit(stage,**kw):
        data=dict(stage=stage,pid=os.getpid(),gpu=a.gpu,seconds=time.time()-start,unix=time.time());data.update(kw)
        event.write(json.dumps(data,allow_nan=False)+'\n');write_json(out/'status.json',data);print(json.dumps(data),flush=True)
    def stop(*_):raise InterruptedError('Bounded probe interrupted; no automatic resume')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def budget():
        if time.time()>=deadline:raise TimeoutError('Short-check time limit reached')
    try:
        emit('initializing');env,cfg=make_pen_simulator(4096,a.gpu,a.seed,a.retarget);dev=env.device;cfg_task=constants(env)
        if a.task_set=='reference':
            env.task_ids[:]=torch.arange(4096,device=dev)%2+2;env.reset_buf.fill_(1);env.step(torch.zeros(4096,22,device=dev))
        write_json(out/'environment.json',cfg);write_json(out/'task-constants.json',cfg_task)
        actor=Actor(DF*H,22).to(dev);model=PenModel(cfg_task,env.pen_tasks).to(dev);task=PenTask(model,cfg_task,dev)
        ss,cc=snapshot(env);dim=task.critic_features(ss,None,None,model.normalize(cc)).shape[-1];critic=Critic(dim).to(dev)
        old_actor=digest(actor.net);old_critic=digest(critic.net)
        if a.algorithm=='sm':
            replay=PenReplay(4096,dev);collector=PenCollector(env,task,actor,replay=replay);err=PhysicalError().to(dev)
            ow=torch.optim.Adam(model.parameters(),lr=args.lr);op=torch.optim.Adam(actor.parameters(),lr=args.lr,betas=(.9,.95));ov=torch.optim.Adam(critic.parameters(),lr=args.lr)
            for _ in range(64):budget();collector.collect(.5)
            valid=replay.epi[:64]>=0;model.fit_stats(replay.s[:64][valid],replay.s_next[:64][valid],replay.ctx[:64][valid]);err.fit(replay.s[:64][valid],replay.s_next[:64][valid])
            emit('collected',valid_rows=collector.valid_rows,parity=collector.max_error)
            for i in range(302 if a.full_warmup else 2):
                budget();fm=fit_update(model,err,replay,ow,args,dev,4096,model.normalize);am={}
                if not a.full_warmup or i>=300:
                    before=digest(model);am=actor_critic_update(actor,critic,model,task,replay,op,ov,args,max(301,i+1),dev,model.normalize,.02,.24)
                    assert digest(model)==before,'Actor update changed frozen physical model'
                    assert am['finite'] and am['gn']>0;collector.actor_updates+=1
                for _ in range(3 if i%2 else 2):collector.collect(.1)
                if i%10==0 or am:emit('update',iteration=i+1,model=fm,actor=am,parity=collector.max_error)
            assert collector.used_actor_updates>=2
            updates=dict(model_optimizer_steps=10*(i+1),actor_updates=collector.actor_updates,critic_optimizer_steps=20*collector.actor_updates)
            checkpoint=dict(actor=actor.checkpoint(),critic=critic.state_dict(),model=model.state_dict(),error=err.state_dict(),config=cfg_task)
        else:
            learner=PairedPPO(actor,critic);collector=PenCollector(env,task,actor,critic=critic,ppo=learner)
            contexts=[]
            for _ in range(64):budget();collector.collect();_,c=snapshot(env);contexts.append(c)
            c=torch.cat(contexts);model.c_mu.copy_(c.mean(0));model.c_sd.copy_(c.std(0).clamp_min(1e-6));del contexts,c
            for i in range(2):
                budget();rows=[collector.collect() for _ in range(32)];metrics=learner.update(rows);collector.actor_updates+=1
                emit('update',iteration=i+1,metrics=metrics,parity=collector.max_error)
            collector.collect();assert collector.used_actor_updates>=2
            updates=dict(actor_updates=2,critic_updates=2,optimizer_steps=learner.steps)
            checkpoint=dict(actor=actor.checkpoint(),critic=critic.state_dict(),context_mean=model.c_mu,context_std=model.c_sd,
                log_std=learner.log_std.detach(),config=cfg_task)
        assert digest(actor.net)!=old_actor and digest(critic.net)!=old_critic
        assert all(torch.isfinite(v).all() for m in (actor,critic,model) for v in m.state_dict().values())
        assert all(file_digest(ROOT/p)==h for p,h in sources.items()),'Hashed source changed during check'
        checkpoint['reference_sampler']=env.reference_sampler.state_dict()
        save_checkpoint(out/'operator.pt',checkpoint)
        result=dict(passed=True,operator_only=not a.full_warmup,full_adaptation_accepted=False,learning_gain_verified=False,
            parity=collector.max_error,valid_rows=collector.valid_rows,used_actor_updates=collector.used_actor_updates,
            critic_dim=dim,seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,**updates)
        write_json(out/'acceptance.json',result);emit('completed',**result)
    except BaseException as exc:emit('failed',error=repr(exc));raise

if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
