"""Bounded first-frame calibration for four complete pen reference candidates."""
import ctypes
import os

ctypes.CDLL('libEGL_nvidia.so.0', mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_NODELETE)
import isaacgym  # noqa: F401

import argparse
import json
from pathlib import Path
import time

import torch

from check_pen_quick_physics import configure_reference
from unionmimic.common import file_digest, write_json
from unionmimic.motions import tracking_metrics
from unionmimic.pen_simulator import make_pen_simulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--fine', action='store_true')
    args = parser.parse_args()
    root, output = args.input.resolve(), args.out.resolve()
    source_hash = file_digest(Path(__file__).resolve())
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    clips, candidates, replicas = ([0, 7], 20, 8) if args.fine else ([0, 2, 6, 7], 24, 4)
    count = len(clips) * candidates * replicas
    seed = 20261246 if args.fine else 20261245
    env, _ = make_pen_simulator(count, args.gpu, seed, root, fixed_phase=-50)
    lower, upper = env.arm_hand_dof_lower_limits, env.arm_hand_dof_upper_limits
    ids = torch.tensor(clips, device=env.device).repeat_interleave(candidates * replicas)
    task = torch.where(ids < 4, 2, 3)
    zeros = torch.zeros(count, 22, device=env.device)
    base_target = env.pen_tasks.target
    best_q = torch.zeros(len(clips), 22, device=env.device)
    best_p = torch.zeros(len(clips), 3, device=env.device)
    history = []

    rounds = (.45,.26,.16) if args.fine else (1.,.55)
    for sigma in rounds:
        q = (best_q[:,None] + torch.randn(len(clips),candidates,22,device=env.device) * .07 * sigma).clamp(-.18,.18)
        p = (best_p[:,None] + torch.randn(len(clips),candidates,3,device=env.device) * .002 * sigma).clamp(-.006,.006)
        q[:,0],p[:,0] = best_q,best_p
        q_offset = q.reshape(-1,22).repeat_interleave(replicas,0)
        p_offset = p.reshape(-1,3).repeat_interleave(replicas,0)

        def target(task_ids, clip_ids, steps):
            assert len(steps) == count
            row = base_target(task_ids,clip_ids,steps)
            following = base_target(task_ids,clip_ids,steps+1)
            weight = ((150-steps.clamp_min(0))/90).clamp(0,1)
            next_weight = ((149-steps.clamp_min(0))/90).clamp(0,1)
            row['q'] = (row['q'] + weight[:,None]*q_offset).clamp(lower,upper)
            following_q = (following['q'] + next_weight[:,None]*q_offset).clamp(lower,upper)
            row['qd'] = torch.where((steps>=0)[:,None], (following_q-row['q'])/.02, row['qd'])
            row['pose'] = torch.cat((row['pose'][:,:3]+p_offset,row['pose'][:,3:]),-1)
            return row

        env.pen_tasks.target = target
        configure_reference(env,task,ids,-50)
        failed = torch.zeros(count,dtype=torch.bool,device=env.device)
        peak = torch.zeros(count,device=env.device)
        error = torch.zeros_like(peak)
        for step in range(100):
            if time.time() > args.deadline:
                raise TimeoutError('Calibration budget expired')
            if step:
                env.reset_buf.zero_()
                env.step(zeros)
            metrics = tracking_metrics(env.local_pose(),target(task,ids,env.reference_step)['pose'])
            failed |= metrics['failure'] | env.pen_failure
            error += metrics['position_error_m']/.05 + metrics['shaft_error_rad']/(torch.pi/3)
            force = env.contact_tensor.view(count,-1,3)[:,env.object_rb_handles].norm(dim=-1).amax(-1)
            peak = torch.maximum(peak,force)
        bad = (failed | (peak>50)).view(len(clips),candidates,replicas)
        score = 10000*bad.sum(-1) + error.view_as(bad).amax(-1) + (peak.view_as(bad)/50).amax(-1).clamp(max=100)
        pick = score.argmin(-1)
        clip_rows = torch.arange(len(clips),device=env.device)
        best_q,best_p = q[clip_rows,pick],p[clip_rows,pick]
        history.append(dict(failed=bad[clip_rows,pick].sum(-1).tolist(),
                            peak_force_n=peak.view_as(bad)[clip_rows,pick].amax(-1).tolist()))
        write_json(output/'calibration-progress.json',history)

    manifest = json.loads((root/'manifest.json').read_text())
    for index, info in enumerate(manifest['clips']):
        source, destination = root/info['file'], output/info['file']
        if index in clips:
            j = clips.index(index)
            saved = torch.load(source,map_location='cpu')
            weight = ((150-torch.arange(len(saved['q'])))/90).clamp(0,1)
            saved['q'] = (saved['q'] + weight[:,None]*best_q[j].cpu()).clamp(lower.cpu(),upper.cpu())
            saved['pose'][:,:3] += best_p[j].cpu()
            saved.pop('actual_links',None)
            torch.save(saved,destination)
        else:
            destination.write_bytes(source.read_bytes())
        info['retarget_sha256'] = file_digest(destination)
    manifest.update(physics_verified=False, quick_lead_calibration=dict(clips=clips, seed=seed, replicas=replicas,
        parent_manifest_sha256=file_digest(root/'manifest.json'), source_sha256=source_hash,
        trials=history))
    write_json(output/'manifest.json',manifest)
    os._exit(0)


if __name__ == '__main__':
    main()
