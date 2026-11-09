"""Learning primitives; no simulator imports. Reference: train_yaw.py @ 0ab03eb.

The physical model is independent of the optional WSM context model (not enabled
in this baseline). Replay windows include terminal outcomes, never reset edges.
"""
import contextlib
import torch
from torch import nn


class BudgetExpired(Exception):
    pass


class RunStopped(Exception):
    pass


class RunningNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        n = len(x)
        if not n:
            return
        total = self.count + n
        delta = x.mean(0) - self.mean
        self.var.copy_((self.var * self.count + x.var(0, unbiased=False) * n
                        + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / (self.var + .01).sqrt()).clamp(-10, 10)


def mlp(din, hidden, dout, layer_norm=False):
    layers = []
    for h in hidden:
        layers.append(nn.Linear(din, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ELU())
        din = h
    layers.append(nn.Linear(din, dout))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim=292, actions=22):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.net = mlp(obs_dim, (512, 256, 128), actions)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        return torch.tanh(self.net(self.norm(obs)))


class Critic(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.norm = RunningNorm(din)
        self.net = mlp(din, (256, 128), 1, True)

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1)


@contextlib.contextmanager
def frozen(*modules):
    params = [p for m in modules for p in m.parameters()]
    previous = [p.requires_grad for p in params]
    for p in params:
        p.grad = None
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(params, previous):
            p.requires_grad_(flag)


def checked_step(loss, optimizer, parameters, value_clip=None):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training objective')
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError('Missing or nonfinite training gradients')
    norm = nn.utils.clip_grad_norm_(parameters, 1.)
    if value_clip is not None:
        nn.utils.clip_grad_value_(parameters, value_clip)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError('Nonfinite parameters after update')
    return float(norm)


def lambda_returns(reward, continuation, values, gamma=.99, lam=.95):
    tail = values[-1]
    result = []
    for t in reversed(range(len(reward))):
        tail = reward[t] + gamma * continuation[t] * ((1 - lam) * values[t + 1] + lam * tail)
        result.append(tail)
    return torch.stack(result[::-1])


def objective(reward, continuation, tail, gamma=.99):
    total, survival = torch.zeros_like(tail), torch.ones_like(tail)
    for t, r in enumerate(reward):
        total = total + gamma ** t * survival * r
        survival = survival * continuation[t]
    return total + gamma ** len(reward) * survival * tail


def gae(reward, value, next_value, failure, timeout, valid, gamma=.99, lam=.95):
    out = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * (~failure[t]).float() * next_value[t] - value[t]
        carry = (delta + gamma * lam * (~(failure[t] | timeout[t])).float() * carry) * valid[t]
        out[t] = carry
    return out, out + value


class Replay:
    def __init__(self, capacity, nenv, device):
        self.capacity, self.nenv, self.device = capacity, nenv, device
        self.data, self.steps = {}, 0

    @torch.no_grad()
    def add(self, row):
        if not self.data:
            self.data = {k: torch.zeros((self.capacity,) + v.shape, dtype=v.dtype,
                                       device=self.device) for k, v in row.items()}
        if set(row) != set(self.data):
            raise ValueError('Replay schema changed')
        for k, v in row.items():
            self.data[k][self.steps % self.capacity].copy_(v)
        self.steps += 1

    def indices(self, batch, window=1, split=None):
        lo, hi = max(0, self.steps - self.capacity), self.steps - window
        if hi < lo:
            raise ValueError('Insufficient replay')
        # Finite search, unlike the reference's unbounded rejection loop.
        gs, es, count = [], [], 0
        for _ in range(100):
            g = torch.randint(lo, hi + 1, (2 * batch,), device=self.device)
            e = torch.randint(self.nenv, (2 * batch,), device=self.device)
            rows = (g[None] + torch.arange(window, device=self.device)[:, None]) % self.capacity
            ep = self.data['episode'][rows, e[None]]
            ok = self.data['valid'][rows, e[None]].all(0) & (ep == ep[:1]).all(0)
            if split=='train':ok &= ep[0]%5!=0
            elif split=='valid':ok &= ep[0]%5==0
            elif split is not None:raise ValueError('Unknown episode split')
            if window > 1:
                ok &= ~(self.data['failure'][rows[:-1], e[None]] |
                        self.data['timeout'][rows[:-1], e[None]]).any(0)
            gs.append(g[ok]); es.append(e[ok]); count += int(ok.sum())
            if count >= batch:
                g, e = torch.cat(gs)[:batch], torch.cat(es)[:batch]
                r = (g[None] + torch.arange(window, device=self.device)[:, None]) % self.capacity
                return {k: v[r, e[None]] for k, v in self.data.items()}
        raise RuntimeError('Replay has too few contiguous valid windows')

    def state_dict(self):
        return dict(capacity=self.capacity, nenv=self.nenv, steps=self.steps,
                    data={k: v.cpu() for k, v in self.data.items()})

    def load_state_dict(self, state):
        if (state['capacity'], state['nenv']) != (self.capacity, self.nenv):
            raise ValueError('Replay dimensions changed on resume')
        self.steps = state['steps']
        self.data = {k: v.to(self.device) for k, v in state['data'].items()}
