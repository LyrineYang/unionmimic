"""Episode replay: row t contains o_t and the incoming transition a_(t-1), r_t.

The first physical step after an automatic reset is an initial observation,
never a transition out of the previous episode's terminal observation.
"""
from .runtime import make_runtime, close_runtime
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.transforms import quaternion_to_matrix
from .common import RELEASE, digest, file_digest, write_json
from .data_contracts import audit_episode


def terminal_flags(env):
    fall = (env.object_pos - env.goal_pos).norm(dim=-1) >= env.fall_dist
    q0, q1 = env.object_init_quat, env.object_rot
    r0 = quaternion_to_matrix(q0[:, [3, 0, 1, 2]])
    r1 = quaternion_to_matrix(q1[:, [3, 0, 1, 2]])
    local_axis = torch.bmm(r0.transpose(1, 2), env.spin_axis[..., None])
    axis = torch.bmm(r1, local_axis).squeeze(-1)
    tilt = torch.acos((axis * env.spin_axis).sum(-1).clamp(-1, 1)) > .4 * np.pi
    stalled = env.no_spin_counter >= env.no_spin_max_steps
    failure = fall | tilt | stalled
    timeout = env.progress_buf >= env.max_episode_length - 1
    assert torch.equal(failure | timeout, env.reset_buf.bool()), 'Unclassified termination'
    return dict(failure=failure, timeout=timeout, fall=fall, tilt=tilt, stalled=stalled)


def collect(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'episodes.h5').exists(), 'Use a new output directory; datasets are immutable'
    gravity=getattr(args,'gravity_max_degrees',None)
    overrides=[] if gravity is None else ['task.env.gravityCurriculum.maxAngleDeg='+str(gravity)]
    env, player, cfg, obs = make_runtime(args.envs, args.gpu, args.seed, args.checkpoint,overrides=overrides)
    wm = env.wm_adapter.world_model
    model_before, actor_before = digest(wm), digest(player.model)
    latest_embed = {}
    hook = wm.encoder.register_forward_hook(lambda m, x, y: latest_embed.update(value=y.detach()))
    buffers = [[] for _ in range(args.envs)]
    episode_count = 0
    records = []
    max_action_error = max_target_error = 0.0
    lower = env.arm_hand_dof_lower_limits.cpu().numpy()
    upper = env.arm_hand_dof_upper_limits.cpu().numpy()
    (out / 'config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    began = time.monotonic()
    try:
        with h5py.File(out / 'episodes.h5', 'w') as f:
            f.attrs['schema_version'] = 1
            f.attrs['source_checkpoint_sha256'] = file_digest(args.checkpoint)
            f.attrs['state_model_sha256'] = digest(wm.encoder) + ':' + digest(wm.dynamics)
            f.attrs['object'] = env.used_training_objects[0]
            f.attrs['seed'] = args.seed
            f.attrs['lower_limits'] = lower
            f.attrs['upper_limits'] = upper
            f.attrs['reward_units'] = 'raw_task_reward'
            f.attrs['alignment'] = 'row[t]: obs_t, incoming executed_action_(t-1), reward_t; first row has no incoming transition'
            for step in range(args.steps):
                reset_before = env.reset_buf.bool().clone()
                command = player.get_action(obs, is_deterministic=True)
                # Separate stress episodes provide physical failures, not synthetic labels.
                stress = torch.arange(args.envs, device=env.device) % args.stress_every == args.stress_every-1
                active = stress & (env.progress_buf > 30)
                if args.probe_mode == 'release':
                    command = torch.where(active[:, None], -torch.ones_like(command), command)
                else:
                    command = (command + active[:, None] * args.probe_noise * torch.randn_like(command)).clamp(-1, 1)
                obs, rewards, dones, _ = player.env_step(env, command)
                flags = terminal_flags(env)
                state = env.wm_adapter._wm_latent
                rows = dict(prop=env.last_obs_buf, depth=obs['depth'], history=env.obs_buf,
                            policy_obs=obs['obs'], command=command, applied_command=env.applied_command,
                            executed_action=env.executed_command, targets=env.prev_targets,
                            last_actions=env.last_actions, scale=env.relative_scale_tensor,
                            reward=env.rew_buf, done=env.reset_buf.bool(), is_first=reset_before,
                            progress=env.progress_buf, embed=latest_embed['value'],
                            post_deter=state['deter'], post_logit=state['logit'],
                            post_index=state['stoch'].argmax(-1).to(torch.uint8), **flags)
                if getattr(args, 'diagnostic_state', False):
                    # Labels only; never provided to the policy, encoder or RSSM.
                    # Hand-local coordinates prevent scene/environment position leakage.
                    from unionmimic.online_baseline.task import qrotate, qconj, qmul
                    obj = env.root_state_tensor[env.object_indices, :13]
                    base = env.root_state_tensor[env.hand_indices, :13]
                    inv = qconj(base[:, 3:7]); local_q = qmul(inv, obj[:, 3:7])
                    xaxis = torch.zeros(args.envs, 3, device=env.device); xaxis[:, 0] = 1
                    yaxis = torch.zeros_like(xaxis); yaxis[:, 1] = 1
                    rows.update(object_position_local=qrotate(inv, obj[:, :3]-base[:, :3]),
                                object_rotation6d=torch.cat((qrotate(local_q,xaxis),qrotate(local_q,yaxis)),-1),
                                object_velocity_local=qrotate(inv, obj[:, 7:10]-base[:, 7:10]))
                rows = {k: v.detach().cpu().numpy().copy() for k, v in rows.items()}
                for i in range(args.envs):
                    if rows['is_first'][i]:
                        assert not buffers[i], 'Previous terminal observation was not finalized'
                    else:
                        assert buffers[i], 'Missing initial observation'
                    buffers[i].append({k: v[i] for k, v in rows.items()})
                    if not rows['done'][i]:
                        continue
                    data = {k: np.stack([row[k] for row in buffers[i]]) for k in rows}
                    buffers[i] = []
                    if len(data['reward']) < 2:
                        continue
                    ae, te = audit_episode(data, lower, upper)
                    max_action_error, max_target_error = max(max_action_error, ae), max(max_target_error, te)
                    key = '%06d' % episode_count
                    group = f.create_group(key)
                    for name, value in data.items():
                        group.create_dataset(name, data=value, compression='lzf', shuffle=True)
                    mode = args.probe_mode if i % args.stress_every == args.stress_every-1 else 'policy'
                    group.attrs['behavior'] = mode
                    group.attrs['env_id'] = i
                    records.append(dict(id=key, env_id=i, behavior=mode,
                                        transitions=len(data['reward'])-1,
                                        failure=bool(data['failure'][-1]),
                                        timeout=bool(data['timeout'][-1]),
                                        fall=bool(data['fall'][-1]),
                                        reward=float(data['reward'][1:].sum())))
                    episode_count += 1
                if (step + 1) % 100 == 0:
                    print(json.dumps(dict(phase='collect', steps=step+1, episodes=episode_count,
                                          seconds=round(time.monotonic()-began, 1))), flush=True)
                    f.flush()
            # Stratification is by complete episode, never by frames/windows.
            rng = np.random.default_rng(args.seed)
            strata = {}
            for r in records:
                strata.setdefault((r['behavior'], r['failure']), []).append(r)
            for group_records in strata.values():
                rng.shuffle(group_records)
                n_valid = max(1, round(len(group_records) * .2)) if len(group_records) >= 3 else 0
                for j, r in enumerate(group_records):
                    r['split'] = 'valid' if j < n_valid else 'train'
                    f[r['id']].attrs['split'] = r['split']
            f.attrs['complete'] = True
            f.attrs['diagnostic_state_labels_only'] = bool(getattr(args, 'diagnostic_state', False))
        assert digest(wm) == model_before and digest(player.model) == actor_before
        report = dict(schema_version=1, episodes=len(records),
                      transitions=sum(r['transitions'] for r in records),
                      failures=sum(r['failure'] for r in records), falls=sum(r['fall'] for r in records),
                      timeouts=sum(r['timeout'] for r in records),
                      train_episodes=sum(r['split']=='train' for r in records),
                      valid_episodes=sum(r['split']=='valid' for r in records),
                      excluded_unfinished_rows=sum(map(len, buffers)),
                      max_action_alignment_error=max_action_error,
                      max_target_alignment_error=max_target_error,
                      parameters_unchanged=True, terminal_observations_preserved=True,
                      seconds=time.monotonic()-began, records=records)
        write_json(out / 'collection.json', report)
        print(json.dumps({k:v for k,v in report.items() if k!='records'}), flush=True)
    finally:
        hook.remove()
        close_runtime(env)
