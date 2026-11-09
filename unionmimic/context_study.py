"""Frozen WSM policy context; physical dynamics and pinned SM updates stay separate.

Diagnostic 10 Hz corner-block interface, not the unfinished 50 Hz pen bridge.
All arms use the same observer for a controlled information experiment. Its cost
is included even for the zero-context arm, so this is not an efficiency benchmark.
"""
import copy
import os
from pathlib import Path
import torch
from torch import nn
import yaml
from in_hand_rotation_mjlab.world_model.policy import RunningNorm, mlp
from isaacgymenvs.utils.utils import set_seed
from unionmimic.common import UPSTREAM, RELEASE
from unionmimic.online_baseline.simulator import StudyRotation, task_from_env
from unionmimic.leap_sharpa.simulator import SharpaReplay, SharpaCollector
from unionmimic.leap_sharpa.model import SharpaTask, X


def make_context_env(nenv, gpu, seed, config):
    torch.cuda.set_device(gpu)
    os.chdir(UPSTREAM)
    cfg = yaml.safe_load(Path(config).read_text())
    cfg = copy.deepcopy(cfg)
    task = cfg['task']; task['env']['numEnvs'] = nenv
    task['env']['isTestRun'] = False
    wcfg = task['env']['cameraPolicy']['worldModel']
    wcfg.update(replay_capacity=2, train_steps_per_epoch=0)
    set_seed(seed, torch_deterministic=True)
    dev = 'cuda:%d' % gpu
    env = StudyRotation(task, dev, dev, gpu, True, False, False)
    assert env.num_actions == 22 and env.n_obs_dim == 73
    assert abs(env.dt * env.control_freq_inv - .10002) < 1e-6
    assert env.control_mode == 'relative' and env.act_moving_average == .8
    assert env.used_training_objects == ['set_obj6_block_corner']
    wm = env.wm_adapter.world_model
    saved = torch.load(str(RELEASE), map_location='cpu')
    wm.load_state_dict(saved['world_model']['world_model'], strict=True)
    wm.eval().requires_grad_(False)
    del saved
    env.study_properties = torch.zeros(nenv, 4, device=dev)
    env.reset_buf[:] = 1
    env.step(torch.zeros(nenv, 22, device=dev))
    assert env.wm_adapter._wm_latent is not None
    native = task_from_env(env)
    native.cfg.update(force_scale=float(env.force_scale), force_probability=float(env.random_force_prob_scalar), force_decay=float(env.force_decay))
    return env, native, cfg


class ContextActor(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ('none', 'fixed', 'rollout')
        self.mode = mode
        self.norm = RunningNorm(730)
        self.context_norm = RunningNorm(512)
        self.project = nn.Linear(512, 64)
        self.net = mlp(794, (512, 512, 256), 22, nn.ELU)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.real_context = None

    def forward(self, obs, noise=0.):
        h = self.real_context if obs.shape[-1] == 730 else obs[:, 730:]
        assert h is not None and h.shape == (len(obs), 512)
        extra = self.project(self.context_norm(h)).tanh()
        if self.mode == 'none':
            extra = extra * 0
        pre = self.net(torch.cat((self.norm(obs[:, :730]), extra), -1))
        if noise > 0:
            pre = pre + noise * torch.randn_like(pre)
        return pre.tanh()

    def checkpoint(self):
        return dict(kind='frozen_context_actor', mode=self.mode,
                    state_dict={k: v.detach().cpu().clone() for k, v in self.state_dict().items()})

    @classmethod
    def from_checkpoint(cls, ck):
        assert ck['kind'] == 'frozen_context_actor'
        actor = cls(ck['mode']); actor.load_state_dict(ck['state_dict'])
        return actor


class ContextTask(SharpaTask):
    def __init__(self, model, constants, device, rssm, mode):
        super().__init__(model, constants, device)
        self.rssm, self.mode, self.context_state = rssm, mode, None
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(7331).get_state()

    def flatten(self, frames):
        assert self.context_state is not None
        return torch.cat((frames.flatten(1), self.context_state['deter']), -1)

    def imagined_frame(self, sn, c):
        frame, sn = super().imagined_frame(sn, c)
        if self.mode == 'rollout':
            # Filtered, executed incremental command, never the raw/residual command.
            # Isolate categorical sampling from the baseline's sensor/action RNG.
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.cuda.set_rng_state(self.rng, self.device)
                self.context_state = self.rssm.img_step(self.context_state, sn[:, X['action']], sample=True)
                self.rng = torch.cuda.get_rng_state(self.device)
        return frame, sn


class ContextReplay(SharpaReplay):
    def __init__(self, n, device, task):
        super().__init__(n, device)
        self.context_task = task
        self.deter = torch.empty(self.L, n, 512, device=device)
        # Categorical posterior samples are exact one-hot indices, not FP caches.
        self.indices = torch.empty(self.L, n, 32, dtype=torch.uint8, device=device)

    @torch.no_grad()
    def cache(self, latent):
        self.deter[self.G % self.L].copy_(latent['deter'])
        self.indices[self.G % self.L].copy_(latent['stoch'].argmax(-1))

    def sample_starts(self, batch, history):
        g, e = super().sample_starts(batch, history)
        self.context_task.context_state = dict(deter=self.deter[g % self.L, e].detach(),
            stoch=torch.nn.functional.one_hot(self.indices[g % self.L, e].long(), 32).float())
        return g, e


class ContextCollector(SharpaCollector):
    @torch.no_grad()
    def collect(self, noise):
        latent = self.env.wm_adapter._wm_latent
        self.replay.cache(latent)
        self.actor.real_context = latent['deter'].detach().clone()
        valid = ~self.env.reset_buf.bool()
        if valid.any():
            self.actor.context_norm.update(self.actor.real_context[valid])
        super().collect(noise)
