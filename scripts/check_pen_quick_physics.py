"""Screen complete reference clips for the bounded first-frame pen experiment."""
import ctypes
import os

ctypes.CDLL('libEGL_nvidia.so.0', mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_NODELETE)
import isaacgym  # noqa: F401

import argparse
from pathlib import Path
import time

import torch

from unionmimic.common import file_digest, write_json
from unionmimic.hand_kinematics import HandKinematics
from unionmimic.motions import tracking_metrics
from unionmimic.motions import qinv, qrot
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.common import ROOT


def configure_reference(env, task, clip_ids, phase):
    env.fixed_clips = True
    env.fixed_phase = phase
    env.task_ids[:] = task
    env.clip_ids.copy_(clip_ids)
    env.reset_buf.fill_(1)
    env.step(torch.zeros(env.num_envs, 22, device=env.device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--task', type=int, choices=(0, 2, 3), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--retarget', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lead-in', action='store_true')
    parser.add_argument('--max-depenetration',type=float)
    parser.add_argument('--velocity-iterations',type=int)
    parser.add_argument('--fixed-friction-scale',type=float)
    args = parser.parse_args()
    retarget, output = args.retarget.resolve(), args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    began = time.monotonic()
    # Four single-axis or nine multi-axis clips, with 32 replicas per clip.
    pool = list(range(13)) if args.task == 0 else (list(range(4)) if args.task == 2 else list(range(4, 13)))
    count = len(pool) * 32
    numerics={} if args.max_depenetration is None else dict(max_depenetration_velocity=args.max_depenetration)
    if args.velocity_iterations is not None:
        numerics['num_velocity_iterations']=args.velocity_iterations
    if args.fixed_friction_scale is not None:
        numerics['fixed_friction_scale']=args.fixed_friction_scale
    env, _ = make_pen_simulator(count, args.gpu, args.seed, retarget, fixed_phase=0, physics_numerics=numerics)
    clips = torch.tensor(pool, device=env.device).repeat(32)
    expected = (torch.cat((env.pen_tasks.single,env.pen_tasks.multi)) if args.task == 0 else
                (env.pen_tasks.single if args.task == 2 else env.pen_tasks.multi))
    assert torch.equal(expected, clips[:len(pool)])
    records = {str(i): {} for i in pool}
    action = torch.zeros(count, 22, device=env.device)
    movement_task = torch.where(clips<4,2,3) if args.task == 0 else args.task
    fk = HandKinematics(ROOT/'work/upstreams/WM-Craftnet/assets/urdf/right_sharpa_wave/right_sharpa_wave.urdf',
                        env.arm_hand_dof_names, env.device)
    names = [f'right_{finger}_fingertip' for finger in ('thumb','index','middle','ring','pinky')]
    handles = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.arm_hands[0], name) for name in names]
    base = env.hand_start_states
    bq = base[:, [6,3,4,5]]
    fk_max = 0.
    prior_good = torch.ones(count, dtype=torch.bool, device=env.device)
    if args.lead_in:
        configure_reference(env, movement_task, clips, -50)
    for mode in ('hold', 'moving'):
        if not args.lead_in:
            configure_reference(env, 1 if mode == 'hold' else movement_task, clips, 0)
        actual = qrot(qinv(bq)[:,None].expand(-1,5,-1),
                      env.rigid_body_states[:,handles,:3]-base[:,None,:3])
        fk_max = max(fk_max, float((actual-fk.forward(env.arm_hand_dof_pos,names)).norm(dim=-1).max()))
        peak = torch.zeros(count, device=env.device)
        failed = torch.zeros(count, dtype=torch.bool, device=env.device)
        first_force = torch.full((count,), -1, device=env.device)
        first_failure = torch.full_like(first_force, -1)
        for step in range(50):
            if step or (args.lead_in and mode == 'moving'):
                env.reset_buf.zero_()
                env.step(action)
            goal = env.pen_tasks.initial[clips] if mode == 'hold' else env.pen_tasks.target(
                env.task_ids, clips, env.reference_step)['pose']
            failure = tracking_metrics(env.local_pose(), goal)['failure'] | env.pen_failure
            force = env.contact_tensor.view(count, -1, 3)[:, env.object_rb_handles].norm(dim=-1).amax(-1)
            assert torch.isfinite(force).all() and torch.isfinite(env.policy_frame).all()
            first_force = torch.where((first_force < 0) & (force > 50), step, first_force)
            first_failure = torch.where((first_failure < 0) & failure, step, first_failure)
            peak = torch.maximum(peak, force)
            failed |= failure
        good = (~failed) & (peak <= 50)
        if args.lead_in:
            good &= prior_good
            prior_good = good
        for index, clip in enumerate(pool):
            rows = slice(index, count, len(pool))
            records[str(clip)][mode] = dict(
                accepted=int(good[rows].sum()), total=32,
                failed=int(failed[rows].sum()), peak_force_n=peak[rows].tolist(),
                first_over_50n_step=first_force[rows].tolist(), first_failure_step=first_failure[rows].tolist())
    write_json(output, dict(task=args.task, seed=args.seed, gpu=args.gpu, seconds=time.monotonic()-began,
                            retarget_sha256=file_digest(retarget/'manifest.json'), clips=records,
                            phase_at_reset=-50 if args.lead_in else 0, fk_max_error_m=fk_max, physics_numerics=numerics))
    os._exit(0)


if __name__ == '__main__':
    main()
