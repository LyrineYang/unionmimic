import math
from pathlib import Path
import torch
from torch import nn
import yaml
from isaacgymenvs.utils.world_model import WorldModelAdapter
from .common import UPSTREAM, RELEASE, digest


def load_world_model(config_path, checkpoint=RELEASE, gpu=0):
    torch.cuda.set_device(gpu)
    cfg = yaml.safe_load(Path(config_path).read_text())
    wm_cfg = dict(cfg['task']['env']['cameraPolicy']['worldModel'])
    wm_cfg.update(replay_capacity=2, train_steps_per_epoch=0)
    adapter = WorldModelAdapter(UPSTREAM, 1, 22, 73, (72, 96, 1), wm_cfg,
                               torch.device('cuda:%d' % gpu))
    saved = torch.load(str(checkpoint), map_location='cpu')
    wm = adapter.world_model
    wm.load_state_dict(saved['world_model']['world_model'], strict=True)
    wm.eval()
    return wm, saved, cfg


def state_model_digest(wm):
    return digest(wm.encoder) + ':' + digest(wm.dynamics)


class CalibratedHeads(nn.Module):
    def __init__(self, wm):
        super().__init__()
        self.wm = wm
        self.failure = nn.Sequential(nn.Linear(wm._feat_size, 256), nn.LayerNorm(256),
                                     nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1))
        self.register_buffer('failure_log_weight', torch.tensor(0.0))

    def freeze_body(self):
        self.wm.requires_grad_(False)
        self.wm.heads['reward'].requires_grad_(True)
        self.wm.heads['decoder']._mlp.requires_grad_(True)
        self.failure.requires_grad_(True)

    def predictions(self, features):
        reward = self.wm.heads['reward'](features)
        prop = self.wm.heads['decoder']._mlp(features)['prop']
        raw_logit = self.failure(features).squeeze(-1)
        return reward, prop, raw_logit - self.failure_log_weight

    def failure_probability(self, features):
        return torch.sigmoid(self.failure(features).squeeze(-1) - self.failure_log_weight)


def state_from_cache(cache, ids, device):
    return {k: cache['state_' + k][ids].to(device) for k in ('deter', 'stoch', 'logit')}
