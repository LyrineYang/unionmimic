import ctypes
import os

# Keep the EGL implementation mapped while its exit-time worker threads finish.
ctypes.CDLL('libEGL_nvidia.so.0', mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_NODELETE)
import isaacgym  # Must precede torch.
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from isaacgymenvs.tasks.wm_craftnet_camera_rotation import RealmanSharpaHa4CameraRotation
from isaacgymenvs.train import _apply_policy_to_rlg_network_cfg
from isaacgymenvs.utils.utils import set_seed
from rl_games.common import env_configurations
from rl_games.algos_torch.players import PpoPlayerContinuous
from .common import UPSTREAM, RELEASE


class RecordedRotation(RealmanSharpaHa4CameraRotation):
    def pre_physics_step(self, actions):
        # This is the command AFTER domain-randomization noise and clipping,
        # BEFORE the native low-pass filter and target integration.
        self.applied_command = actions.detach().clone()
        super().pre_physics_step(actions)
        self.executed_command = self.actions.detach().clone()


def make_runtime(num_envs=16, gpu=0, seed=42, checkpoint=RELEASE, overrides=None):
    torch.cuda.set_device(gpu)
    os.chdir(UPSTREAM)
    with initialize_config_dir(config_dir=str(UPSTREAM / 'isaacgymenvs/cfg'), version_base='1.1'):
        runtime_overrides = [
            'task=WMCraftnetRotation', 'train=WMCraftnetPPO',
            'headless=True', 'test=True', 'wandb_activate=False',
            'capture_video=False', 'force_render=False',
            'sim_device=cuda:%d' % gpu, 'rl_device=cuda:%d' % gpu,
            'graphics_device_id=%d' % gpu, 'seed=%d' % seed,
            'train.params.tuning.num_envs=%d' % num_envs,
            'train.params.tuning.minibatch_size=%d' % num_envs,
            'task.env.numEnvs=%d' % num_envs,
            'task.env.objSet=set_z', 'task.env.axis=z',
            '++task.env.singleObject=set_obj6_block_corner',
            '++task.env.isTestRun=False', 'task.env.evalMetrics.enabled=False',
            'task.env.cameraDemo.enabled=False', 'task.env.cameraDemo.inference_video.enabled=False',
            'task.env.wmDepthVideoRecord.enabled=False', 'task.env.spinTrace.enabled=False',
            '++task.env.wmFeatureRecord.enabled=False',
            'task.env.cameraPolicy.worldModel.wm_prop_pred=True',
            'task.env.cameraPolicy.worldModel.wm_depth_pred=True',
            'task.env.cameraPolicy.worldModel.wm_pose_pred=False',
            'task.env.cameraPolicy.worldModel.wm_tac_pred=False',
            'task.env.cameraPolicy.worldModel.wm_value_pred=False',
            'task.env.cameraPolicy.worldModel.wm_obj_pred=False',
            'task.env.cameraPolicy.worldModel.replay_capacity=8',
            'task.env.cameraPolicy.worldModel.train_steps_per_epoch=0',
        ]
        runtime_overrides.extend(overrides or [])
        cfg = compose(config_name='config', overrides=runtime_overrides)
    set_seed(seed, torch_deterministic=True)
    env = RecordedRotation(OmegaConf.to_container(cfg.task, resolve=True),
                           cfg.rl_device, cfg.sim_device, gpu, True, False, False)
    assert env.used_training_objects == ['set_obj6_block_corner']
    env_configurations.register('unionmimic', {'env_creator': lambda **kw: env})
    params = OmegaConf.to_container(cfg.train, resolve=True)
    _apply_policy_to_rlg_network_cfg(cfg, params)
    params['params']['config'].update(env_name='unionmimic', device_name='cuda:%d' % gpu)
    player = PpoPlayerContinuous(params['params'])
    player.restore(str(checkpoint))
    saved = torch.load(str(checkpoint), map_location='cpu')
    env.wm_adapter.world_model.load_state_dict(saved['world_model']['world_model'], strict=True)
    del saved
    env.wm_adapter.world_model.eval()
    player.model.eval()
    obs = player.env_reset(env)
    player.batch_size = player.get_batch_size(obs, num_envs)
    player.init_rnn()
    return env, player, cfg, obs


def close_runtime(env):
    if os.environ.get('UNIONMIMIC_PROCESS_OWNS_SIM')=='1':
        # CLI workers have closed their own output files already. Synchronize
        # CUDA, then let process exit reclaim native resources together instead
        # of racing destroy_sim against renderer/driver finalizer threads.
        torch.cuda.synchronize(env.device)
        return
    env.close()
    env.gym.destroy_sim(env.sim)
