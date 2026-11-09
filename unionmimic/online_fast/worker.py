"""Bounded worker with atomic learner+replay checkpoints and offline-first logs."""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from unionmimic.common import ROOT, digest, file_digest, save_checkpoint, write_json
from unionmimic.online_baseline.core import Actor, Critic, Replay, BudgetExpired, RunStopped
from unionmimic.online_baseline.task import PhysicalModel, DS, DC
from unionmimic.online_baseline.simulator import make_simulator, Collector
from .learner import Learner
from unionmimic.online_screen.branch import collection_steps
from unionmimic.online_baseline.tracking import tracking_run_id


DEFAULTS=dict(envs=64,batch=256,model_horizon=8,horizon=8,model_updates=10,
              actor_lr=2e-4,model_lr=2e-4,critic_lr=2e-4,ppo_lr=1e-4,
              gamma=.99,**{'lambda':.95},critic_epochs=5,ppo_batch=512,
              collect_steps=8,ppo_steps=32,warmup_steps=128,warmup_updates=200,
              replay_steps=512,exploration=.1)
DEFAULTS.update(failure_fraction=0., critic_norm_source='mixed')


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(),python=random.getstate())


def restore_rng(r):
    torch.set_rng_state(r['torch'].cpu());torch.cuda.set_rng_state(r['cuda'].cpu())
    np.random.set_state(r['numpy']);random.setstate(r['python'])


def sources():
    files=list((ROOT/'unionmimic/online_fast').glob('*.py'))
    files+=list((ROOT/'unionmimic/online_baseline').glob('*.py'))
    files+=[ROOT/'scripts/train_online_fast.py',ROOT/'unionmimic/online_screen/branch.py',ROOT/'unionmimic/common.py']
    files += [ROOT/'scripts/env.sh',ROOT/'config/runtime-lock.json',ROOT/'config/wandb.env']
    files += [ROOT/'work/upstreams/WM-Craftnet'/p for p in [
        'isaacgymenvs/tasks/wm_craftnet_rotation.py','isaacgymenvs/tasks/wm_craftnet_camera_rotation.py',
        'isaacgymenvs/tasks/base/vec_task.py','rl_games/common/common_losses.py']]
    return {str(p.relative_to(ROOT)):file_digest(p) for p in files}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--arm',choices=['sm','ppo'],required=True)
    parser.add_argument('--gpu',type=int,default=1);parser.add_argument('--seed',type=int,default=92001)
    parser.add_argument('--minutes',type=float,default=15);parser.add_argument('--cycles',type=int,default=40)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--config')
    parser.add_argument('--eval-episodes',type=int,default=0);parser.add_argument('--checkpoint')
    args=parser.parse_args()
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    cfg={**DEFAULTS}
    if args.config:cfg.update(json.loads(Path(args.config).read_text()))
    start=time.time();deadline=min(start+args.minutes*60,cfg.get('deadline_unix',float('inf')))
    status_path=run/'status.json'
    if args.resume:
        previous=json.loads(status_path.read_text())
        if previous['status'] in ['stopped','failed']:
            raise RuntimeError('User-stopped/failed workers require explicit diagnosis, not automatic resume')
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest['sources']!=sources() or manifest['config']!=cfg or manifest['arm']!=args.arm:
            raise RuntimeError('Resume source/config mismatch')
        deadline=manifest['deadline_unix']
        if time.time()>=deadline:raise RuntimeError('Original deadline expired')
    else:
        if (run/'manifest.json').exists():raise FileExistsError('Use a new immutable run directory')
        manifest=dict(started_unix=start,deadline_unix=deadline,arm=args.arm,seed=args.seed,gpu=args.gpu,
                      config=cfg,sources=sources(),released_actor_loaded=False,prior_loaded=False,
                      long_training_approved=bool(cfg.get('launch_authorization')),
                      launch_authorization=cfg.get('launch_authorization'),evaluation=bool(args.eval_episodes))
        write_json(run/'manifest.json',manifest)
        for path in manifest['sources']:
            dest=run/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((ROOT/path).read_bytes())
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    log=(run/'events.jsonl').open('a',buffering=1)
    tb=None;wb=None;learner=None;collector=None
    outcome=['initializing']
    def emit(kind,**values):
        row=dict(kind=kind,unix=time.time(),**values)
        log.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
        if tb:
            for k,v in values.items():
                if isinstance(v,(float,int)):tb.add_scalar(kind+'/'+k,v,learner.counts['cycles'])
        if wb:wb.log({kind+'/'+k:v for k,v in values.items() if isinstance(v,(float,int))})
    def status(value,phase,**extra):
        outcome[0]=value
        write_json(status_path,dict(status=value,phase=phase,pid=os.getpid(),gpu=args.gpu,arm=args.arm,
            updated_unix=time.time(),started_unix=manifest['started_unix'],deadline_unix=deadline,
            counts=learner.counts if learner else {},frames=collector.frames if collector else 0,**extra))
    def checkpoint():
        save_checkpoint(run/'last.pt',dict(schema=1,learner=learner.state_dict(),replay=replay.state_dict(),
            rng=rng_state(),collector=collector.snapshot(),manifest_sha256=file_digest(run/'manifest.json'),
            task_constants=task.cfg,initial_actor_digest=initial_actor,initial_critic_digest=initial_critic,
            sources=manifest['sources'],parity=collector.evidence(),config=cfg))
    try:
        status('running','initialize')
        if cfg.get('allocator_fraction'):
            torch.cuda.set_device(args.gpu)
            torch.cuda.set_per_process_memory_fraction(cfg['allocator_fraction'],args.gpu)
        env,task,sim_cfg=make_simulator(cfg['envs'],args.gpu,args.seed)
        write_json(run/'sim-config.json',sim_cfg)
        # Same actor and critic RNG sequence in both arms, independent of model creation.
        torch.manual_seed(args.seed+100000)
        actor=Actor().to(env.device);critic=Critic(DS+DC).to(env.device)
        initial_actor,initial_critic=digest(actor),digest(critic)
        model=PhysicalModel(task).to(env.device)
        learner=Learner(actor,critic,model,task,cfg)
        learner.deadline=deadline
        learner.stop_requested=lambda:stop[0]
        replay=Replay(cfg['replay_steps'],cfg['envs'],env.device)
        collector=Collector(env,task,actor,critic)
        from torch.utils.tensorboard import SummaryWriter
        tb=SummaryWriter(str(run/'tensorboard'))
        import wandb
        tracking_id=tracking_run_id(run,args.arm)
        wb=wandb.init(project=os.environ.get('WANDB_PROJECT','unionmimic'),
            entity=os.environ.get('WANDB_ENTITY'),id=tracking_id,resume='allow',
            name=tracking_id,dir=str(run),config=dict(arm=args.arm,seed=args.seed,**cfg),
            mode=os.environ.get('WANDB_MODE','offline'),
            settings=wandb.Settings(init_timeout=30,disable_git=True,disable_code=True))
        if args.resume:
            saved=torch.load(run/'last.pt',map_location=env.device)
            assert saved['manifest_sha256']==file_digest(run/'manifest.json')
            learner.load_state_dict(saved['learner']);replay.load_state_dict(saved['replay'])
            collector.frames=saved['collector']['frames']
            offset=saved['collector']['next_episode']+cfg['envs']
            collector.episode+=offset;collector.next_episode+=offset
            initial_actor=saved['initial_actor_digest'];initial_critic=saved['initial_critic_digest']
            restore_rng(saved['rng'])
            restored=collector.restore_curriculum(saved['collector']['curriculum'])
            emit('resume',cycles=learner.counts['cycles'],frames=collector.frames,
                 replay_steps=replay.steps,new_simulator_episodes=True,
                 curriculum_restored=True,curriculum_angle=restored['values']['gravity_curriculum_current_max_angle_deg'],
                 curriculum_count=restored['values']['gravity_curriculum_outcome_count'])
        elif args.eval_episodes:
            if args.checkpoint:
                saved=torch.load(args.checkpoint,map_location=env.device)
                actor.load_state_dict(saved['learner']['actor']);critic.load_state_dict(saved['learner']['critic'])
            while len(collector.completed)<args.eval_episodes and time.time()<deadline and not stop[0]:
                collector.step(noise=0,normalize=False,actor_version=0)
            if len(collector.completed)<args.eval_episodes:raise RuntimeError('Incomplete physical evaluation')
            episodes=collector.completed[:args.eval_episodes]
            write_json(run/'evaluation.json',dict(episodes=episodes,
                mean_return=float(np.mean([x['return'] for x in episodes])),
                mean_rotation=float(np.mean([x['rotation'] for x in episodes])),
                mean_axis_deviation=float(np.mean([x['axis_deviation_rad'] for x in episodes])),
                mean_offaxis=float(np.mean([x['offaxis'] for x in episodes])),
                failures=sum(x['failure'] for x in episodes),parity=collector.evidence()))
            status('completed','evaluate');return
        else:
            emit('initialization',actor_digest=initial_actor,critic_digest=initial_critic,
                 policy_from_scratch=True,physical_state_dim=DS,context_dim=DC)
            save_checkpoint(run/'initial.pt',dict(learner=learner.state_dict(),config=cfg,
                task_constants=task.cfg,actor_digest=initial_actor,critic_digest=initial_critic))
            for step in range(cfg['warmup_steps']):
                learner.check_budget()
                replay.add(collector.step(args.arm,noise=.5,log_std=learner.log_std,actor_version=0))
            b=replay.indices(4096,split='train')
            model.fit_stats(b['s'][0],b['sn'][0],b['ctx'][0])
            if args.arm=='sm':
                for j in range(cfg['warmup_updates']):
                    learner.check_budget()
                    # Optional reference-style online model-only warm-up. Actor
                    # remains random; collect fresh exploration, never teacher data.
                    if cfg.get('warmup_collect_steps',0) and j % cfg['model_updates']==0:
                        for _ in range(cfg['warmup_collect_steps']):
                            learner.check_budget()
                            replay.add(collector.step(args.arm,noise=.5,log_std=learner.log_std,actor_version=0))
                    report=learner.fit(replay,1)
                    if (j+1)%25==0:emit('warmup',updates=j+1,**report);status('running','warmup')
            checkpoint()
        episodes_written=0;last_save=time.time();first_cycle=learner.counts['cycles']
        while learner.counts['cycles']-first_cycle<args.cycles and time.time()<deadline and not stop[0]:
            tic=time.time();rows=[];version=learner.counts['actor']
            n=collection_steps(cfg,learner.counts['cycles']) if args.arm=='sm' else cfg['ppo_steps']
            status('running','collect')
            for _ in range(n):
                learner.check_budget()
                row=collector.step(args.arm,noise=cfg['exploration'],log_std=learner.log_std,
                                   normalize=(args.arm=='sm'),actor_version=version)
                replay.add(row);rows.append(row)
            status('running','update')
            if args.arm=='sm':
                report=learner.fit(replay);report.update(learner.imagine(replay))
                if learner.counts['cycles']%10==0:report.update(learner.validate_model(replay))
            else:report=learner.ppo(rows)
            emit('cycle',cycle=learner.counts['cycles'],frames=collector.frames,
                 model_updates=learner.counts['model'],actor_updates=learner.counts['actor'],
                 critic_updates=learner.counts['critic'],seconds=time.time()-tic,
                 collected_actor_version=version,collected_vector_steps=n,**report)
            recent=collector.completed[episodes_written:]
            if recent:
                emit('physical_episodes',episodes=len(recent),
                     mean_return=float(np.mean([x['return'] for x in recent])),
                     mean_rotation=float(np.mean([x['rotation'] for x in recent])),
                     mean_axis_deviation=float(np.mean([x['axis_deviation_rad'] for x in recent])),
                     mean_offaxis=float(np.mean([x['offaxis'] for x in recent])),
                     failure_rate=float(np.mean([x['failure'] for x in recent])))
            with (run/'episodes.jsonl').open('a') as f:
                for ep in collector.completed[episodes_written:]:f.write(json.dumps(ep)+'\n')
            episodes_written=len(collector.completed)
            if time.time()-last_save>120:
                checkpoint();last_save=time.time()
        checkpoint()
        write_json(run/'acceptance.json',dict(counts=learner.counts,frames=collector.frames,
            initial_actor_digest=initial_actor,final_actor_digest=digest(actor),
            initial_critic_digest=initial_critic,final_critic_digest=digest(critic),
            original_actor_loaded=False,independent_model=True,replay_steps=replay.steps,
            parity=collector.evidence(),complete_episodes=len(collector.completed),
            resume_semantics=collector.snapshot()['resume_semantics'],
            seconds=time.time()-start,cuda_peak_allocated=torch.cuda.max_memory_allocated()))
        # Cycle-bounded pilots can explicitly resume within their original deadline.
        status('stopped' if stop[0] else ('completed' if time.time()>=deadline else 'segment_complete'),'finished')
    except BudgetExpired:
        checkpoint()
        status('completed','deadline',partial_cycle_preserved=True)
    except RunStopped:
        checkpoint()
        status('stopped','user_signal',partial_cycle_preserved=True)
    except BaseException as exc:
        status('failed','exception',error=repr(exc))
        raise
    finally:
        if tb:tb.close()
        if wb:
            wb.summary['worker_status']=outcome[0]
            wb.finish(exit_code=1 if outcome[0]=='failed' else 0)
        log.close()

