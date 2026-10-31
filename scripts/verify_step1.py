#!/usr/bin/env python3
"""Run bounded checks against real CUDA, Isaac Gym and the published policy.

Each stage runs in its own process. A native teardown crash remains a failure.
No policy or world-model training is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "work/upstreams/WM-Craftnet"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare_graphics_runtime():
    # Driver 530.30.02 unloads eglcore while its [vkrt] Analysis thread is
    # still running during normal process exit. Keep the EGL library mapped;
    # simulator/camera cleanup and normal Python/native exit still run.
    import ctypes
    ctypes.CDLL("libEGL_nvidia.so.0", mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_NODELETE)


def cuda_check(args):
    import torch
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(args.gpu)
    x = torch.randn(128, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize()
    require(x.grad is not None and bool(torch.isfinite(x.grad).all()), "Invalid CUDA gradient")
    require(float(x.grad.abs().sum()) > 0, "Zero CUDA gradient")
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.gpu),
            "capability": list(torch.cuda.get_device_capability(args.gpu)),
            "gradient_norm": float(x.grad.norm())}


def gym_check(args):
    prepare_graphics_runtime()
    from isaacgym import gymapi, gymtorch
    import torch
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1.0 / 60.0
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    sim = gym.create_sim(args.gpu, args.gpu, gymapi.SIM_PHYSX, params)
    require(sim is not None, "Isaac Gym GPU sim creation failed")
    try:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane)
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
        asset = gym.create_box(sim, .15, .15, .15, gymapi.AssetOptions())
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, .6)
        gym.create_actor(env, asset, pose, "probe_box", 0, 0)
        camera_props = gymapi.CameraProperties()
        camera_props.width, camera_props.height = 96, 72
        camera_props.enable_tensors = True
        camera = gym.create_camera_sensor(env, camera_props)
        require(camera >= 0, "GPU camera creation failed")
        gym.set_camera_location(camera, env, gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, .2))
        gym.prepare_sim(sim)
        roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        contacts = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        image_tensor = gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(sim, env, camera, gymapi.IMAGE_DEPTH))
        require(roots.is_cuda and image_tensor.is_cuda, "Physics/camera tensors are not on GPU")
        min_z, max_contact = .6, 0.0
        for _ in range(100):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_net_contact_force_tensor(sim)
            require(bool(torch.isfinite(roots).all()), "Nonfinite physics state")
            min_z = min(min_z, float(roots[0, 2]))
            max_contact = max(max_contact, float(contacts.norm(dim=-1).max()))
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)
        depth = image_tensor.clone()
        gym.end_access_image_tensors(sim)
        valid = torch.isfinite(depth) & (depth < 0)
        require(min_z < .3, "Gravity did not move the probe box")
        require(max_contact > 0, "Ground contact did not produce a force")
        require(int(valid.sum()) > 100, "Camera has no valid scene depth")
        result = {"physics_steps": 100, "minimum_box_z": min_z,
                  "maximum_contact_force": max_contact, "depth_shape": list(depth.shape),
                  "valid_depth_pixels": int(valid.sum()), "depth_device": str(depth.device)}
    finally:
        gym.destroy_sim(sim)
    return result


def module_digest(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def policy_check(args):
    prepare_graphics_runtime()
    import isaacgym  # Must precede torch imports.
    import torch
    import numpy as np
    from PIL import Image
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import isaacgymenvs
    from isaacgymenvs.train import _apply_policy_to_rlg_network_cfg
    from isaacgymenvs.utils.utils import set_seed
    from rl_games.common import env_configurations
    from rl_games.algos_torch.players import PpoPlayerContinuous

    checkpoint_path = ROOT / "work/checkpoints/wm_craftnet_set_z.pth"
    require(checkpoint_path.is_file(), "Published z checkpoint is missing")
    require(sha256(checkpoint_path) == "3215205fe1fad48e97f5745ab90bb2377d53012c1a0d2634edf186699ccab7f7",
            "Checkpoint differs from the pinned, downloaded release")
    require(args.steps >= 200, "Policy verification requires at least 200 control steps")
    os.chdir(UPSTREAM)
    with initialize_config_dir(config_dir=str(UPSTREAM / "isaacgymenvs/cfg"), version_base="1.1"):
        cfg = compose(config_name="config", overrides=[
            "task=WMCraftnetRotation", "train=WMCraftnetPPO", "headless=True", "test=True",
            "wandb_activate=False", "capture_video=False", "force_render=False",
            "sim_device=cuda:%d" % args.gpu, "rl_device=cuda:%d" % args.gpu,
            "graphics_device_id=%d" % args.gpu, "seed=%d" % args.seed,
            "train.params.tuning.num_envs=1", "train.params.tuning.minibatch_size=1",
            "task.env.numEnvs=1", "task.env.objSet=set_z", "task.env.axis=z",
            "task.env.resetPositionNoise=[0.03,0.015,0.001]", "task.env.resetRotationNoise=5.0",
            "++task.env.isTestRun=False", "task.env.evalMetrics.enabled=False",
            "task.env.cameraDemo.enabled=False", "task.env.cameraDemo.inference_video.enabled=False",
            "task.env.wmDepthVideoRecord.enabled=False", "task.env.spinTrace.enabled=False",
            "++task.env.wmFeatureRecord.enabled=False",
            "task.env.cameraPolicy.worldModel.wm_prop_pred=True",
            "task.env.cameraPolicy.worldModel.wm_depth_pred=True",
            "task.env.cameraPolicy.worldModel.wm_pose_pred=False",
            "task.env.cameraPolicy.worldModel.wm_tac_pred=False",
            "task.env.cameraPolicy.worldModel.wm_value_pred=False",
            "task.env.cameraPolicy.worldModel.wm_obj_pred=False",
            "task.env.cameraPolicy.worldModel.replay_capacity=8",
            "task.env.cameraPolicy.worldModel.train_steps_per_epoch=0",
        ])
    set_seed(args.seed, torch_deterministic=True)
    (args.output / "policy-config.yaml").write_text(OmegaConf.to_yaml(cfg))
    env = isaacgymenvs.make(cfg.seed, cfg.task_name, 1, cfg.sim_device, cfg.rl_device,
                           cfg.graphics_device_id, True, False, False, False, cfg)
    try:
        env_configurations.register("unionmimic_verify", {"env_creator": lambda **kw: env})
        config = OmegaConf.to_container(cfg.train, resolve=True)
        _apply_policy_to_rlg_network_cfg(cfg, config)
        config["params"]["config"].update({"env_name": "unionmimic_verify",
                                            "device_name": "cuda:%d" % args.gpu})
        player = PpoPlayerContinuous(config["params"])
        player.restore(str(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        env.wm_adapter.world_model.load_state_dict(checkpoint["world_model"]["world_model"], strict=True)
        del checkpoint
        wm = env.wm_adapter.world_model
        wm.eval()
        player.model.eval()
        def forbid_training(*unused, **kw):
            raise RuntimeError("World-model training called during inference-only verification")
        wm._train = forbid_training
        wm._model_opt._opt.step = forbid_training
        before = {"policy": module_digest(player.model), "world_model": module_digest(wm)}
        obs = player.env_reset(env)
        player.batch_size = player.get_batch_size(obs, 1)
        player.init_rnn()
        require(bool(env.sensor_valid_mask.all()), "A fingertip contact sensor has no rigid-body handle")
        rewards, done_count, contact_switches = [], 0, 0
        previous_contact, maximum_contact = None, 0.0
        initial_q = env.arm_hand_dof_pos.clone()
        depth_shapes = set()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(args.gpu)
        for step in range(args.steps):
            action = player.get_action(obs, is_deterministic=True)
            require(tuple(action.shape) == (1, 22), "Unexpected Sharpa action shape")
            require(bool(torch.isfinite(action).all()), "Nonfinite policy action")
            obs, reward, done, _ = player.env_step(env, action)
            for name in ("obs", "depth", "wm_feature"):
                require(bool(torch.isfinite(obs[name]).all()), "Nonfinite " + name)
            require(obs["depth"].is_cuda, "Policy depth is not on GPU")
            require(float(obs["depth"].std()) > 1e-5, "Policy camera has a constant image")
            depth_shapes.add(tuple(obs["depth"].shape))
            sensed = env.sensed_contacts
            require(tuple(sensed.shape) == (1, 5), "Unexpected fingertip contact shape")
            require(bool(((sensed == 0) | (sensed == 1)).all()), "Invalid binary tactile values")
            force = env.contact_tensor.view(1, -1, 3)[:, env.sensor_handle_indices].norm(dim=-1)
            force = force * env.sensor_valid_mask.unsqueeze(0)
            expected_contact = (force >= env.tac_thresh).float()
            require(bool(torch.equal(expected_contact, env.gt_contacts)), "Touch does not match measured force")
            maximum_contact = max(maximum_contact, float(force.max()))
            if previous_contact is not None:
                contact_switches += int((previous_contact != env.gt_contacts).sum())
            previous_contact = env.gt_contacts.clone()
            require(bool(torch.isfinite(reward).all()), "Nonfinite task reward")
            rewards.append(float(reward.mean()))
            done_count += int(done.sum())
            if step in (0, args.steps - 1):
                depth = obs["depth"][0, ..., 0].detach().cpu().numpy()
                Image.fromarray((np.clip(depth, 0, 1) * 255).astype("uint8")).save(
                    args.output / ("depth-%04d.png" % step))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        require(maximum_contact > 0, "No fingertip contact response was observed")
        q_change = float((env.arm_hand_dof_pos - initial_q).abs().max())
        require(q_change > 1e-6, "Policy commands did not move the hand")
        # Explicit reset probe after measured rollout, so it does not affect reward totals.
        reset_ids = torch.arange(1, device=env.device)
        env.reset_idx(reset_ids, reset_ids)
        require(int(env.progress_buf.max()) == 0, "Episode progress did not reset")
        require(bool(env.wm_adapter._wm_is_first.all()), "RSSM was not marked for reset")
        reset_obs, _, _, _ = player.env_step(env, torch.zeros((1, 22), device=env.device))
        require(bool(torch.isfinite(reset_obs["wm_feature"]).all()), "RSSM reset failed")
        require(int(env.progress_buf.max()) == 1, "Simulation did not advance after reset")
        require(not bool(env.wm_adapter._wm_is_first.any()), "RSSM reset was not consumed")
        after = {"policy": module_digest(player.model), "world_model": module_digest(wm)}
        require(before == after, "Inference changed policy or world-model parameters/buffers")
        result = {"checkpoint_sha256": sha256(checkpoint_path), "seed": args.seed,
                  "control_steps": args.steps, "num_envs": 1, "object": env.used_training_objects[0],
                  "reward_sum": sum(rewards), "reward_mean": sum(rewards) / len(rewards),
                  "episode_terminations": done_count, "explicit_reset_passed": True,
                  "max_joint_displacement": q_change, "max_fingertip_force": maximum_contact,
                  "contact_switches": contact_switches, "depth_shapes": [list(s) for s in depth_shapes],
                  "elapsed_seconds": elapsed, "peak_torch_memory_bytes": torch.cuda.max_memory_allocated(args.gpu),
                  "parameters_unchanged": before == after, "parameter_sha256": before}
    finally:
        env.close()
        env.gym.destroy_sim(env.sim)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "cuda", "gym", "policy"), default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "work/results/step1")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        summary = {}
        for stage in ("cuda", "gym", "policy"):
            with open(args.output / (stage + ".log"), "w") as log:
                cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                       "--gpu", str(args.gpu), "--steps", str(args.steps), "--seed", str(args.seed),
                       "--output", str(args.output)]
                run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            result_path = args.output / (stage + ".json")
            result = json.loads(result_path.read_text()) if result_path.exists() else {"passed": False}
            result["returncode"] = run.returncode
            result["passed"] = bool(result.get("passed")) and run.returncode == 0
            summary[stage] = result
            print(stage, "PASS" if result["passed"] else "FAIL", flush=True)
        summary["passed"] = all(summary[s]["passed"] for s in ("cuda", "gym", "policy"))
        summary["provenance"] = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "verifier_sha256": sha256(Path(__file__).resolve()),
            "upstream_revision": subprocess.check_output(
                ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
            "upstream_status": subprocess.check_output(
                ["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip(),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return 0 if summary["passed"] else 1
    path = args.output / (args.stage + ".json")
    # Overwrite any old success before starting a stage that may crash natively.
    path.write_text(json.dumps({"passed": False, "status": "running"}) + "\n")
    try:
        result = {"passed": True, **{"cuda": cuda_check, "gym": gym_check, "policy": policy_check}[args.stage](args)}
    except Exception as exc:
        result = {"passed": False, "error": str(exc), "traceback": traceback.format_exc()}
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
