"""Compact observable geometry and a calibrated imagination error model.

The readout receives only causal history and frozen encoder output. Simulator
positions are supervision labels and SM-predicted states, never deployment input.
"""
import torch
from torch import nn


class PositionReadout(nn.Module):
    def __init__(self, mean, scale, weight, offset, use_embed=True):
        super().__init__()
        for name, tensor in [('mean', mean), ('scale', scale), ('weight', weight), ('offset', offset)]:
            self.register_buffer(name, tensor.float().detach().clone())
        self.use_embed = use_embed

    def forward(self, history, embed=None):
        history = history.flatten(1)
        if self.use_embed:
            assert embed is not None
            features = torch.cat((history, embed), -1)
        else:
            features = history
        assert features.shape[-1] == len(self.mean)
        return ((features-self.mean)/self.scale).clamp(-10, 10) @ self.weight + self.offset

    def checkpoint(self, **metadata):
        return dict(kind='hand_local_position_readout', use_embed=self.use_embed,
                    state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()}, **metadata)

    @classmethod
    def from_checkpoint(cls, ck):
        assert ck['kind'] == 'hand_local_position_readout'
        s=ck['state_dict']
        return cls(s['mean'],s['scale'],s['weight'],s['offset'],ck['use_embed'])


class PositionErrorBridge(nn.Module):
    """Conditional AR(1) residual, fitted on disjoint calibration trajectories.

This is a candidate observation bridge, not a physics model or calibrated risk
score. Sampling remains differentiable with respect to SM's predicted position.
"""
    def __init__(self, mean, rho, innovation_cholesky):
        super().__init__()
        for name, value in [('mean', mean), ('rho', rho), ('innovation_cholesky', innovation_cholesky)]:
            self.register_buffer(name, value.float().detach().clone())

    def step(self, predicted_position, previous_error, noise):
        assert predicted_position.shape == previous_error.shape == noise.shape
        error = self.mean + (previous_error-self.mean)*self.rho + noise @ self.innovation_cholesky.T
        return predicted_position+error, error

    def checkpoint(self, **metadata):
        return dict(kind='position_error_ar1',state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},**metadata)

    @classmethod
    def from_checkpoint(cls, ck):
        assert ck['kind']=='position_error_ar1';s=ck['state_dict']
        return cls(s['mean'],s['rho'],s['innovation_cholesky'])
