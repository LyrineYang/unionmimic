"""Bounded, fresh-policy context ablation using the pinned LEAP update functions."""
import ctypes, os, sys
ctypes.CDLL('libEGL_nvidia.so.0', mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'work/runtime/leap-sharpa/src'), str(ROOT/'work/runtime/leap-sharpa')]
import argparse, json, time, traceback, signal, subprocess
from types import SimpleNamespace
import torch
from in_hand_rotation_mjlab.world_model.policy import Critic
from native_update import fit_update, actor_critic_update
from unionmimic.common import write_json, file_digest, digest, save_checkpoint, RELEASE
from unionmimic.context_study import make_context_env, ContextActor, ContextTask, ContextReplay, ContextCollector
from unionmimic.leap_sharpa.model import SharpaModel, PhysicalError


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True); p.add_argument('--run', required=True)
    p.add_argument('--mode', choices=['none', 'fixed', 'rollout'], required=True)
    p.add_argument('--rounds', type=int, default=400); p.add_argument('--operator-only', action='store_true')
    p.add_argument('--envs', type=int, default=256); p.add_argument('--seed', type=int, default=20261212)
    p.add_argument('--max-seconds', type=int, default=5400)
    cli = p.parse_args(); run = Path(cli.run).resolve(); run.mkdir(parents=True, exist_ok=False)
    config = ROOT/'work/analysis/sm-context-trajectories-20261109-v1/config.yaml'
    started = time.time(); deadline = min(started + cli.max_seconds, 1794311095.)
    assert deadline > started and (cli.operator_only or cli.rounds > 300)
    args = SimpleNamespace(wm_batch=2048, wm_steps=10, wm_k=8, starts=4096, horizon=16,
        wm_only=300, utd=.4, warmup=64, lr=2e-4, gamma=.99, lam=.95, imag_noise=0., imag_noise_final=None, imag_noise_updates=6000)
    sources = [Path(__file__).resolve(), ROOT/'scripts/evaluate_context_study.py', ROOT/'unionmimic/context_study.py',
               ROOT/'unionmimic/online_baseline/simulator.py']
    sources += list((ROOT/'unionmimic/leap_sharpa').glob('*.py')) + list((ROOT/'work/runtime/leap-sharpa').rglob('*.py'))
    hashes = {str(s.relative_to(ROOT)):file_digest(s) for s in sources}
    write_json(run/'manifest.json', dict(started_unix=started, deadline_unix=deadline, cli=vars(cli), recipe=vars(args),
        sources=hashes, wsm_sha256=file_digest(RELEASE), actor_from_scratch=True, critic_from_scratch=True,
        observer_all_arms=True, scope='10Hz camera block mechanism experiment, 256 envs instead of reference 4096; NOT reproduction/pen/efficiency benchmark'))
    for src in sources:
        target = run/'source'/src.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(src.read_bytes())
    log = (run/'events.jsonl').open('a', buffering=1); updates = 0
    def emit(kind, **kw):
        row = dict(kind=kind, unix=time.time(), seconds=time.time()-started, update=updates, **kw)
        log.write(json.dumps(row, allow_nan=False)+'\n'); print(json.dumps(row, allow_nan=False), flush=True)
    def status(value, **kw):
        write_json(run/'status.json', dict(status=value, pid=os.getpid(), gpu=cli.gpu, mode=cli.mode, updates=updates,
            actor_updates=max(0,updates-300), updated_unix=time.time(), started_unix=started, deadline_unix=deadline, **kw))
    def stop(*_): raise InterruptedError('Stopped; no automatic resume')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        status('running', phase='init'); torch.set_num_threads(4)
        env, native, cfg = make_context_env(cli.envs, cli.gpu, cli.seed, config); dev = env.device
        # Reset learner RNG after identical sensor/env construction in every arm.
        torch.manual_seed(cli.seed + 1)
        actor = ContextActor(cli.mode).to(dev); wm = SharpaModel(native.cfg).to(dev); err = PhysicalError().to(dev)
        prior = env.wm_adapter.world_model
        task = ContextTask(wm, native.cfg, dev, prior.dynamics, cli.mode)
        replay = ContextReplay(cli.envs, dev, task)
        critic = Critic(task.critic_features(torch.zeros(2,177,device=dev),None,None,torch.zeros(2,100,device=dev)).shape[-1]).to(dev)
        opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
        opt_pi = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(.9,.95))
        opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
        initial = dict(actor=digest(actor.net), critic=digest(critic.net), model=digest(wm), prior=digest(prior))
        write_json(run/'native-config.json', cfg); write_json(run/'initial.json', initial)
        collector = ContextCollector(env, native, actor, replay)
        def save(name):
            save_checkpoint(run/name, dict(actor=actor.checkpoint(), wm=wm.state_dict(), critic=critic.state_dict(), error=err.state_dict(),
                updates=updates, optimizers=[o.state_dict() for o in (opt_wm,opt_pi,opt_v)], task_constants=native.cfg,
                prior_sha256=file_digest(RELEASE), exact_resume=False))
        save('initial.pt')
        for _ in range(64): collector.collect(.5)
        valid = replay.epi[:64] >= 0
        wm.fit_stats(replay.s[:64][valid], replay.s_next[:64][valid], replay.ctx[:64][valid])
        err.fit(replay.s[:64][valid], replay.s_next[:64][valid]); del valid
        emit('initialized', parity=collector.max_error, valid_rows=collector.valid_rows)
        if cli.operator_only:
            fm = fit_update(wm,err,replay,opt_wm,args,dev,cli.envs,wm.normalize)
            frozen = digest(wm)
            am = actor_critic_update(actor,critic,wm,task,replay,opt_pi,opt_v,args,301,dev,wm.normalize,native.cfg['control_dt'],task.max_step_reward())
            assert digest(wm)==frozen and digest(prior)==initial['prior']
            assert digest(actor.net)!=initial['actor'] and digest(critic.net)!=initial['critic'] and am['gn']>0
            assert all(p.grad is None and not p.requires_grad for p in prior.parameters())
            collector.actor_updates=1
            for _ in range(8): collector.collect(.1)
            write_json(run/'result.json',dict(passed=True,operator_only=True,model=fm,actor=am,parity=collector.max_error,
                prior_unchanged=True,model_frozen_in_actor=True,recollection=collector.used_actor_updates))
            status('completed'); return
        acc=0.; totals=dict(fit=0.,actor=0.,collect=0.)
        while updates < cli.rounds:
            if time.time()>deadline-60: raise TimeoutError('Original bounded study budget exhausted')
            t=time.time(); collector.collect(.1); totals['collect']+=time.time()-t; acc+=args.utd
            if acc<1: continue
            acc-=1; updates+=1
            t=time.time(); fm=fit_update(wm,err,replay,opt_wm,args,dev,cli.envs,wm.normalize); totals['fit']+=time.time()-t
            am={}
            if updates>300:
                t=time.time()
                am=actor_critic_update(actor,critic,wm,task,replay,opt_pi,opt_v,args,updates,dev,wm.normalize,native.cfg['control_dt'],task.max_step_reward())
                totals['actor']+=time.time()-t; collector.actor_updates=updates-300
            if updates%10==0:
                status('running', phase='warmup' if updates<=300 else 'policy', frames=replay.G*cli.envs)
                emit('update',**fm,**am,frames=replay.G*cli.envs,peak_gib=torch.cuda.max_memory_allocated()/1024**3)
            if updates in (300,350,cli.rounds): save('round-%04d.pt'%updates)
        for _ in range(3): collector.collect(.1)
        assert digest(prior)==initial['prior'] and all(p.grad is None and not p.requires_grad for p in prior.parameters())
        assert collector.used_actor_updates==cli.rounds-300
        assert all(file_digest(ROOT/s)==h for s,h in hashes.items()), 'Sources changed during experiment'
        for m in (actor,critic,wm):
            assert all(torch.isfinite(v).all() for v in m.state_dict().values())
        save('final.pt')
        result=dict(passed=True,scope='short mechanism control; not reproduction or established benefit',updates=updates,
            actor_updates=updates-300,model_updates=updates*10,critic_updates=(updates-300)*20,frames=replay.G*cli.envs,
            parity=collector.max_error,prior_unchanged=True,initial=initial,
            final=dict(actor=digest(actor.net),critic=digest(critic.net),model=digest(wm)),
            totals_seconds=totals,seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        write_json(run/'result.json',result); status('completed'); emit('completed',result=result)
    except BaseException as exc:
        status('stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc)); raise

if __name__=='__main__':
    rc=0
    try: main()
    except BaseException: traceback.print_exc(); rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
