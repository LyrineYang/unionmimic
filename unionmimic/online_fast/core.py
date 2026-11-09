"""Same numerical safety checks, aggregated before host synchronization."""
import torch
from torch import nn
from unionmimic.online_baseline.core import frozen, lambda_returns, objective, gae, BudgetExpired, RunStopped


def finite_group(tensors):
    # All entries remain checked. One scalar host read per group replaces one
    # host read per tensor; loss, clipping, optimizer and RNG are unchanged.
    return bool(torch.stack([torch.isfinite(x).all() for x in tensors]).all())


def checked_step(loss, optimizer, parameters, value_clip=None):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training objective')
    parameters=list(parameters)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    grads=[p.grad for p in parameters if p.grad is not None]
    if not grads or not finite_group(grads):
        raise FloatingPointError('Missing or nonfinite training gradients')
    norm=nn.utils.clip_grad_norm_(parameters,1.)
    if value_clip is not None:nn.utils.clip_grad_value_(parameters,value_clip)
    optimizer.step()
    if not finite_group(parameters):
        raise FloatingPointError('Nonfinite parameters after update')
    return float(norm)
