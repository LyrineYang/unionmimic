#!/usr/bin/env python3
"""Bounded 4096-env Sharpa adapter check. No learner or training launch."""
import ctypes
import os
import sys
ctypes.CDLL('libEGL_nvidia.so.0', mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_NODELETE)
import isaacgym  # noqa: F401 -- before torch
import torch
import argparse
import json
import signal
import time
import traceback
from pathlib import Path

from unionmimic.common import ROOT, write_json, file_digest
from unionmimic.online_baseline.simulator import make_simulator, state, context
from unionmimic.online_baseline.task import S
from unionmimic.sharpa_leap import L, physical_state, object_root_from_physical, SharpaCommandMap

sys.path.insert(0, str(ROOT / 'work/upstreams/sm-reference-audit/src'))
from in_hand_rotation_mjlab.world_model.policy import Actor  # unchanged reference class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    if run.exists():raise FileExistsError(run)
    run.mkdir(parents=True)
    started = time.time()
    def status(value, **kw):
        write_json(run/'status.json', dict(status=value, updated_unix=time.time(), pid=os.getpid(), **kw))
    def stop(*_):raise InterruptedError('Adapter check interrupted; no automatic resume')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    paths = ['unionmimic/sharpa_leap.py', 'scripts/check_sharpa_leap_physics.py',
             'unionmimic/online_baseline/simulator.py', 'unionmimic/online_baseline/task.py',
             'work/upstreams/sm-reference-audit/src/in_hand_rotation_mjlab/world_model/policy.py',
             'work/upstreams/WM-Craftnet/isaacgymenvs/tasks/wm_craftnet_rotation.py',
             'work/upstreams/WM-Craftnet/isaacgymenvs/tasks/base/vec_task.py']
    write_json(run/'manifest.json', dict(started_unix=started, envs=4096, gpu=a.gpu,
        training=False, learner_integrated=False, long_training_approved=False,
        actor_source='pinned LEAP policy.Actor', actor_hidden=[512,512,256],
        actor_history_frames=10, native_frame_dim=73, policy_or_prior_loaded=False,
        steps=80, recorded_steps=64, deadline_unix=started+600,
        timeout_fixture='step70: advance 16 valid env progress counters; after recorded learning data',
        sources={p:file_digest(ROOT/p) for p in paths}))
    try:
        torch.set_num_threads(4)
        status('running', phase='initialize')
        env, task, config = make_simulator(4096, a.gpu, 20261103)
        assert list(env.actuated_dof_indices.cpu().numpy()) == list(range(22))
        assert env.clip_actions == 1, 'Controller adapter requires native clip [-1,1]'
        controller = SharpaCommandMap(task.cfg['lower'], task.cfg['upper']).to(env.device)
        actor = Actor(73*10, 22).to(env.device)
        # Chronological frames; retain native touch and target/axis features.
        frame = env.obs_buf[:, :73].clamp(-5,5).clone()
        history = frame[:,None].expand(-1,10,-1).clone()
        episode = torch.arange(env.num_envs, device=env.device)
        next_episode = env.num_envs
        rows = []
        maximum = dict(action=0., target=0., reward=0., position=0., velocity=0., quat_sign=0.)
        valid_rows = failures = timeouts = resets = 0
        loop_start = time.time()
        for t in range(80):
            if time.time()-started>600:raise TimeoutError('Adapter check wall budget exceeded')
            if t==70:
                ids=(~env.reset_buf.bool()).nonzero().flatten()[:16]
                env.progress_buf[ids]=env.max_episode_length-2
            invalid = env.reset_buf.bool().clone()
            if invalid.any():
                ids = invalid.nonzero().flatten()
                episode[ids] = torch.arange(next_episode,next_episode+len(ids),device=env.device)
                next_episode += len(ids)
            s = state(env);ctx = context(env)
            base = env.root_state_tensor[env.hand_indices,:13].clone()
            physical = physical_state(s[:,S['q']],s[:,S['v']],s[:,S['target']],s[:,S['object']],base)
            with torch.no_grad():
                valid = ~invalid
                if valid.any():actor.norm.update(history[valid].flatten(1))
                command = (actor(history.flatten(1)) + (.5 if t<64 else .1)*torch.randn(env.num_envs,22,device=env.device)).clamp(-1,1)
                env.step(command)
                sn = state(env)
                predicted_target, executed = controller(s[:,S['target']],env.study_command,
                                                       s[:,S['action']],env.relative_scale_tensor)
                pn = physical_state(sn[:,S['q']],sn[:,S['v']],sn[:,S['target']],sn[:,S['object']],base)
                recovered = object_root_from_physical(pn,base)
                truth = sn[:,S['object']]
                maximum['position'] = max(maximum['position'],float((recovered[:,:3]-truth[:,:3]).abs().max()))
                maximum['velocity'] = max(maximum['velocity'],float((recovered[:,7:]-truth[:,7:]).abs().max()))
                # Compare rotations modulo quaternion sign, without unstable acos at identity.
                q1,q2 = recovered[:,3:7],truth[:,3:7]
                qe = torch.minimum((q1-q2).abs().amax(-1),(q1+q2).abs().amax(-1))
                maximum['quat_sign'] = max(maximum['quat_sign'],float(qe.max()))
                reward,failure,timeout,_,_ = task.reward(s,sn,ctx)
                if valid.any():
                    maximum['action'] = max(maximum['action'],float((executed[valid]-env.actions[valid]).abs().max()))
                    maximum['target'] = max(maximum['target'],float((predicted_target[valid]-env.prev_targets[valid]).abs().max()))
                    maximum['reward'] = max(maximum['reward'],float((reward[valid]-env.rew_buf[valid]).abs().max()))
                    assert torch.equal((failure|timeout)[valid],env.reset_buf.bool()[valid]), 'Terminal before reset mismatch'
                for tensor in (physical,pn,ctx,env.rew_buf,env.obs_buf):
                    assert torch.isfinite(tensor).all(), 'Nonfinite physical/observation/label'
                done = env.reset_buf.bool().clone()
                failures += int((failure&valid).sum());timeouts += int((timeout&valid).sum())
                resets += int(invalid.sum());valid_rows += int(valid.sum())
                if t<64:
                    rows.append({k:v.detach().cpu().clone() for k,v in dict(
                        s=physical,sn=pn,ctx=ctx,command=env.study_command,
                        last_action=s[:,S['action']],executed=executed,scale=env.relative_scale_tensor,
                        frame=frame,valid=valid,episode=episode,done=done,failure=failure,
                        timeout=timeout,base=base[:,:7],reward=env.rew_buf).items()})
                frame=env.obs_buf[:,:73].clamp(-5,5).clone()
                history=torch.cat((history[:,1:],frame[:,None]),1)
                # Reset rows did not have a pre-action observation from their new episode.
                # Exclude that transition; initialize subsequent history only from new frames.
                history[invalid]=frame[invalid,None].expand(-1,10,-1)
            if t%16==15:
                status('running',phase='physics',vector_steps=t+1,valid_rows=valid_rows)
                print(json.dumps(dict(vector_steps=t+1,seconds=time.time()-loop_start,max_error=maximum)),flush=True)
        assert maximum['action']<=1e-6 and maximum['target']<=1e-6
        assert maximum['reward']<=2e-3
        assert max(maximum[k] for k in ('position','velocity','quat_sign'))<=1e-5
        assert valid_rows>0 and resets>0 and timeouts>0, 'No valid/reset/timeout evidence'
        dataset={k:torch.stack([row[k] for row in rows]) for k in rows[0]}
        torch.save(dict(data=dataset,task_constants=task.cfg,joint_names=env.arm_hand_dof_names,
                        schema='sharpa-leap-physical-v1'),run/'transitions.pt')
        write_json(run/'native_config.json',config)
        result=dict(interface_passed=True,full_sm_loop_passed=False,learning_gain_verified=False,
            envs=env.num_envs,vector_steps=80,recorded_vector_steps=64,valid_rows=valid_rows,
            failures=failures,timeouts=timeouts,incoming_reset_rows=resets,max_error=maximum,
            control_dt=task.cfg['control_dt'],actor_history_frames=10,observation_dim=730,
            actor_hidden=list(actor.hidden),actor_weights_updated=False,
            elapsed_seconds=time.time()-started,physics_seconds=time.time()-loop_start,
            torch_peak_allocated_gib=torch.cuda.max_memory_allocated(a.gpu)/1024**3,
            limitations=['Contact/reward prediction heads not implemented',
                         'No F fitting/BPTT/critic updates in this interface check',
                         'No online cross-runtime transport or resumable learner'])
        # Check the recorded source identity again before accepting output.
        manifest=json.loads((run/'manifest.json').read_text())
        result['sources_unchanged']=all(file_digest(ROOT/p)==h for p,h in manifest['sources'].items())
        assert result['sources_unchanged']
        write_json(run/'acceptance.json',result)
        status('completed',phase='finished',interface_passed=True)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        status('stopped' if isinstance(exc,InterruptedError) else 'failed',phase='check',error=repr(exc))
        raise


if __name__ == '__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    # Native destruction can segfault. Propagate all task errors before this exit.
    sys.stdout.flush();sys.stderr.flush()
    os._exit(code)
