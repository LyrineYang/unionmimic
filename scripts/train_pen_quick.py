"""Train one verified pen reference or segment within a short budget."""
import ctypes
import os

ctypes.CDLL('libEGL_nvidia.so.0', mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_NODELETE)
import isaacgym  # noqa: F401

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'work/runtime/leap-sharpa'
sys.path[:0] = [str(RUNTIME / 'src'), str(RUNTIME)]

import torch
from in_hand_rotation_mjlab.world_model.policy import Actor, Critic
from native_update import actor_critic_update, fit_update

from check_pen_quick_physics import configure_reference
from unionmimic.common import digest, file_digest, save_checkpoint, write_json
from unionmimic.leap_sharpa.model import PhysicalError
from unionmimic.leap_sharpa.pen_collect import PenCollector, snapshot
from unionmimic.leap_sharpa.pen_model import DF, H, PenTask, constants
from unionmimic.pen_quick import QuickPenModel, QuickPenReplay
from unionmimic.pen_simulator import make_pen_simulator


def verified_reference(auth, evidence_dir, task, retarget, numerics):
    manifest = json.loads((retarget / 'manifest.json').read_text())
    retarget_hash = file_digest(retarget / 'manifest.json')
    evidence = []
    for seed in auth['physics_seeds']:
        path = evidence_dir / f'physics-task0-seed{seed}.json'
        receipt = json.loads(path.read_text())
        assert (receipt['task'], receipt['seed'], receipt['phase_at_reset']) == (0, seed, -50)
        assert receipt['fk_max_error_m'] < 1e-4
        assert receipt['retarget_sha256'] == retarget_hash
        assert receipt.get('physics_numerics',{}) == numerics
        evidence.append(receipt['clips'])
    passing = [
        int(clip) for clip in evidence[0]
        if (int(clip)<4) == (task==2) and all(
            row['total'] == 32 and row['accepted'] / row['total'] >= .9
            for case in evidence for row in (case[clip]['hold'], case[clip]['moving'])
        )
    ]
    if not passing:
        raise RuntimeError('No clip passed both first-frame physical checks on both seeds')
    return min(passing, key=lambda clip: (manifest['clips'][clip]['frames'], clip)), retarget_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--task', type=int, choices=(2, 3), required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--run', type=Path, required=True)
    cli = parser.parse_args()
    auth_path, run = cli.authorization.resolve(), cli.run.resolve()
    auth = json.loads(auth_path.read_text())
    assert auth['user_authorized'] and auth['scope'] == 'bounded_first_frame_pen_prototype'
    deadline = auth['deadline_unix']
    assert 0 < deadline - auth['started_unix'] <= 1800
    assert auth['original_full_gate_unchanged'] and not auth['auto_resume']
    assert cli.gpu in range(4)
    retarget = (ROOT / auth['retarget_candidates'][cli.candidate]).resolve()
    evidence_dir = auth_path.parent / f'lead-{cli.candidate}'
    numerics = auth.get('numerics_by_candidate',{}).get(cli.candidate,{})
    clip, retarget_hash = verified_reference(auth, evidence_dir, cli.task, retarget, numerics)
    started = time.time()
    if deadline - started < 540:
        raise TimeoutError('Original budget cannot cover training and final evaluation')
    run.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(envs=512, wm_batch=512, wm_steps=5, wm_k=8, starts=512, horizon=16,
                           wm_only=24, utd=.4, warmup=64, warmup_noise=.15, lr=2e-4, gamma=.99, lam=.95,
                           imag_noise=0., imag_noise_final=None, imag_noise_updates=6000)
    paths = [
        Path(__file__).resolve(), ROOT/'scripts/check_pen_quick_physics.py', ROOT/'scripts/evaluate_pen_policy.py',
        ROOT/'scripts/calibrate_pen_quick_lead.py',
        ROOT/'assets/reference_pen/pen_tracking220.urdf', ROOT/'unionmimic/hand_kinematics.py',
        ROOT/'unionmimic/motions.py', ROOT/'unionmimic/sharpa_leap.py', ROOT/'unionmimic/online_baseline/task.py',
        ROOT/'unionmimic/online_baseline/simulator.py',
    ]
    paths += list((ROOT/'unionmimic').glob('pen*.py'))
    paths += list((ROOT/'unionmimic/leap_sharpa').glob('*.py')) + list(RUNTIME.rglob('*.py'))
    paths += [ROOT/'work/upstreams/WM-Craftnet/isaacgymenvs/tasks'/name for name in (
        'wm_craftnet_rotation.py','wm_craftnet_camera_rotation.py','base/vec_task.py')]
    sources = {str(path.relative_to(ROOT)): file_digest(path) for path in paths}
    for path in paths:
        destination = run/'source'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    write_json(run/'manifest.json', dict(task=cli.task, clip=clip, gpu=cli.gpu, from_scratch=True,
        recipe=vars(args), started_unix=started, deadline_unix=deadline, authorization=auth,
        retarget_sha256=retarget_hash, retarget=str(retarget), physics_evidence=str(evidence_dir),
        reference_scope='original_rate_segment' if cli.candidate=='segment' else 'whole_clip',
        physics_numerics=numerics, fixed_phase=-50, full_adaptation_accepted=False, sources=sources))
    iteration = actor_updates = 0
    collector = None
    log = (run/'events.jsonl').open('a', buffering=1)

    def emit(stage, **fields):
        row = dict(stage=stage, gpu=cli.gpu, pid=os.getpid(), unix=time.time(), elapsed=time.time()-started,
                   iteration=iteration, actor_updates=actor_updates, **fields)
        log.write(json.dumps(row, allow_nan=False) + '\n')
        write_json(run/'status.json', row)
        print(json.dumps(row), flush=True)

    def stop(*_):
        raise InterruptedError('Bounded run stopped')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        emit('initializing', clip=clip)
        torch.set_num_threads(4)
        env, config = make_pen_simulator(args.envs, cli.gpu, auth['training_seed'], retarget, fixed_phase=-50,
                                         physics_numerics=numerics)
        device = env.device
        configure_reference(env, cli.task, torch.full_like(env.clip_ids, clip), -50)
        cfg = constants(env)
        write_json(run/'environment.json', config)
        model = QuickPenModel(cfg, env.pen_tasks).to(device)
        task = PenTask(model, cfg, device)
        actor = Actor(DF*H, 22).to(device)
        s, c = snapshot(env)
        critic = Critic(task.critic_features(s, None, None, model.normalize(c)).shape[-1]).to(device)
        del s, c
        error = PhysicalError().to(device)
        replay = QuickPenReplay(args.envs, device)
        collector = PenCollector(env, task, actor, replay=replay)
        ow = torch.optim.Adam(model.parameters(), lr=args.lr)
        op = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(.9, .95))
        ov = torch.optim.Adam(critic.parameters(), lr=args.lr)
        initial = (digest(actor.net), digest(critic.net))
        for _ in range(args.warmup):
            collector.collect(args.warmup_noise)
        valid = replay.epi[:args.warmup] >= 0
        model.fit_stats(replay.s[:args.warmup][valid], replay.s_next[:args.warmup][valid], replay.ctx[:args.warmup][valid])
        error.fit(replay.s[:args.warmup][valid], replay.s_next[:args.warmup][valid])
        emit('collected', valid_rows=collector.valid_rows)

        def save(name):
            assert all(torch.isfinite(v).all() for module in (actor, critic, model) for v in module.state_dict().values())
            save_checkpoint(run/name, dict(actor=actor.checkpoint(), critic=critic.state_dict(), model=model.state_dict(),
                optimizers=[opt.state_dict() for opt in (ow, op, ov)], error=error.state_dict(), iteration=iteration,
                actor_updates=actor_updates, task=cli.task, clip=clip, retarget_manifest_sha256=retarget_hash, physics_numerics=numerics,
                exact_simulator_resume=False, replay_included=False))

        credit = 0.
        last_saved = time.time()
        while time.time() < deadline - 300:
            collector.collect(.1)
            credit += args.utd
            if credit < 1:
                continue
            credit -= 1
            iteration += 1
            metrics = fit_update(model, error, replay, ow, args, device, args.envs, model.normalize)
            if iteration > args.wm_only:
                before = digest(model) if not actor_updates else None
                update = actor_critic_update(actor, critic, model, task, replay, op, ov, args, iteration,
                                             device, model.normalize, .02, task.max_step_reward())
                if before is not None:
                    assert digest(model) == before and update['finite'] and update['gn'] > 0.
                metrics.update(update)
                actor_updates += 1
                collector.actor_updates = actor_updates
            if iteration % 10 == 0 or actor_updates in (1, 2):
                emit('training' if actor_updates else 'model_warmup', valid_rows=collector.valid_rows,
                     used_actor_updates=collector.used_actor_updates, metrics=metrics, parity=collector.max_error)
            if actor_updates and time.time() - last_saved >= 120:
                save('latest.pt')
                last_saved = time.time()
        assert collector.used_actor_updates >= 2 and all(h != digest(net) for h, net in zip(initial, (actor.net, critic.net)))
        save('final.pt')
        emit('evaluating', valid_rows=collector.valid_rows)
        eval_command = [
            sys.executable, str(ROOT/'scripts/evaluate_pen_policy.py'), '--gpu', str(cli.gpu),
            '--checkpoint', str(run/'final.pt'), '--retarget', str(retarget),
            '--single-task', str(cli.task), '--single-clip', str(clip), '--seed', str(auth['evaluation_seed']), '--envs', '128',
        ]
        with (run/'evaluation-final.log').open('w') as stream:
            subprocess.run(eval_command+['--out', str(run/'evaluation-final.json')], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=deadline-time.time()-2)
        result = json.loads((run/'evaluation-final.json').read_text())
        assert result['passed'] and result['episodes'] == 128 and result['single_clip'] == clip
        assert result['checkpoint_sha256'] == file_digest(run/'final.pt')
        if deadline-time.time() >= 115:
            with (run/'evaluation-zero-residual.log').open('w') as stream:
                try:
                    subprocess.run(eval_command+['--out', str(run/'evaluation-zero-residual.json'), '--zero-residual'],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=deadline-time.time()-2)
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                    emit('zero_residual_eval_incomplete', error=repr(error))
        assert all(file_digest(ROOT / name) == value for name, value in sources.items())
        emit('completed', summary=result['summary'], valid_rows=collector.valid_rows)
    except BaseException as error:
        emit('stopped' if isinstance(error, InterruptedError) else 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
