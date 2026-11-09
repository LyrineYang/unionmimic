"""Sharpa physical/controller interface for the pinned LEAP model.

This is an adapter, not a learner. Contact/reward heads and the online learner
integration are not implemented here. Commands are AFTER native action noise
and clipping; never feed the raw policy output as if noise had been applied.
"""
import torch
from torch import nn

from unionmimic.online_baseline.task import qmul, qconj, qrotate


NJ = 22
PHYSICAL_DIM = 3 * NJ + 13
# Same field order as pinned LEAP record.state_slices(nj=22).
L = dict(q=slice(0, 22), qd=slice(22, 44), cmd=slice(44, 66),
         p=slice(66, 69), quat=slice(69, 73), v=slice(73, 76), w=slice(76, 79))


def physical_state(q, qd, target, object_root, base_root):
    """World/xyzw Isaac state -> fixed-hand frame/wxyz LEAP physical state.

    Isaac root linear AND angular velocities are world-frame quantities.
    The current task has a fixed base; moving-base transport is unsupported.
    """
    if q.shape[-1] != NJ or qd.shape != q.shape or target.shape != q.shape:
        raise ValueError('Expected the native ordered 22-joint state and target')
    if object_root.shape != q.shape[:-1] + (13,) or base_root.shape != object_root.shape:
        raise ValueError('Expected paired actor-root states with 13 columns')
    if not torch.isfinite(base_root).all() or base_root[..., 7:].abs().max() > 1e-5:
        raise ValueError('Only a finite, fixed hand base is supported')
    bq = base_root[..., 3:7]
    if (bq.norm(dim=-1) - 1).abs().max() > 1e-4:
        raise ValueError('Hand base quaternion is not normalized')
    inv = qconj(bq)
    pos = qrotate(inv, object_root[..., :3] - base_root[..., :3])
    quat = qmul(inv, object_root[..., 3:7])
    quat = torch.where(quat[..., 3:] < 0, -quat, quat)
    wxyz = torch.cat((quat[..., 3:], quat[..., :3]), -1)
    return torch.cat((q, qd, target, pos, wxyz,
                      qrotate(inv, object_root[..., 7:10]),
                      qrotate(inv, object_root[..., 10:13])), -1)


def object_root_from_physical(s, base_root):
    """Inverse transform for auditing and native reward geometry, not sensing."""
    if s.shape[-1] != PHYSICAL_DIM:
        raise ValueError('Expected LEAP physical state with nj=22')
    bq = base_root[..., 3:7]
    quat = s[..., L['quat']]
    xyzw = torch.cat((quat[..., 1:], quat[..., :1]), -1)
    return torch.cat((qrotate(bq, s[..., L['p']]) + base_root[..., :3],
                      qmul(bq, xyzw), qrotate(bq, s[..., L['v']]),
                      qrotate(bq, s[..., L['w']])), -1)


class SharpaCommandMap(nn.Module):
    """Native 0.8/0.2 action filter and accumulated, limited joint target.

    last_action is explicit recurrent controller memory. It cannot be dropped
    simply because LEAP's physical vector only stores the previous target.
    scale is the actual per-step randomized native scale during parity checks.
    """
    def __init__(self, lower, upper):
        super().__init__()
        self.register_buffer('lower', torch.as_tensor(lower, dtype=torch.float32))
        self.register_buffer('upper', torch.as_tensor(upper, dtype=torch.float32))
        if self.lower.shape != (NJ,) or self.upper.shape != (NJ,):
            raise ValueError('Expected native 22-joint limits')

    def forward(self, target, command, last_action, scale):
        executed = .8 * command.clamp(-1, 1) + .2 * last_action
        target_next = (target + scale * executed).maximum(self.lower).minimum(self.upper)
        return target_next, executed
