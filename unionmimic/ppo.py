"""PPO actor loss from pinned WM-Craftnet/rl_games, with an explicit latent critic.

The critic matches SM's architecture/input. We adapt rollout boundaries and
weight-only warm start; we do not implement a replacement PPO loss.
"""
import copy
import time
from pathlib import Path
import gym
import numpy as np
import torch
from torch import nn
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.tr_helpers import DefaultRewardsShaper
from .sm import critic_input


def make_critic(wm, device):
    return nn.Sequential(nn.Linear(wm._feat_size + 292 + 44, 256), nn.LayerNorm(256),
                         nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1)).to(device)


def gaussian_kl(mu, sigma, reference_mu, reference_sigma):
    if not (torch.isfinite(sigma).all() and torch.isfinite(reference_sigma).all()
            and (sigma>0).all() and (reference_sigma>0).all()):
        raise FloatingPointError('Invalid PPO Gaussian standard deviation')
    p=torch.distributions.Normal(mu,sigma)
    q=torch.distributions.Normal(reference_mu,reference_sigma)
    return torch.distributions.kl_divergence(p,q).sum(-1).mean().clamp_min(0.)


def generalized_advantage(reward, value, next_value, failure, timeout, valid, gamma=.99, lam=.95):
    """Bootstrap time limits from the terminal observation, never a reset row.

    Shapes: [time, env]. Invalid incoming-reset transitions break recursion and
    are excluded from both actor and critic minibatches. A true failure wins
    when a failure and time limit coincide.
    """
    assert reward.shape == value.shape == next_value.shape == failure.shape == timeout.shape == valid.shape
    advantage = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * (~failure[t]).float() * next_value[t] - value[t]
        carry = (delta + gamma * lam * (~(failure[t] | timeout[t])).float() * carry) * valid[t]
        advantage[t] = carry
    return advantage, advantage + value


class AdaptedPPO(A2CAgent):
    """Only the optimizer boundary differs from upstream's actor update."""
    def __init__(self, params, dry_run=True, approval=None):
        self.dry_run = dry_run
        self.approval = approval
        self.optimizer_steps = 0
        self.last_grad_norm = None
        super().__init__('unionmimic-ppo', params)
        # SM's independent latent critic supplies GAE; do not fit the released
        # actor network's unused/shared value head as a second critic.
        self.has_value_loss = False

    def set_train(self):
        super().set_train()
        self.model.running_mean_std.eval()
        self.model.value_mean_std.eval()

    def calc_gradients(self, input_dict):
        super().calc_gradients(input_dict)
        # The pinned upstream adds 1e-5 to variances, reporting negative KL
        # even for identical small-sigma distributions. Correct only the KL
        # measurement used by the LR scheduler; keep its PPO loss unchanged.
        values=list(self.train_result);self.upstream_kl=float(values[3])
        values[3]=gaussian_kl(values[6],values[7],input_dict['mu'],input_dict['sigma'])
        self.train_result=tuple(values)

    def trancate_gradients_and_step(self):
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        assert parameters and all(torch.isfinite(p.grad).all() for p in parameters)
        self.last_grad_norm = float(nn.utils.clip_grad_norm_(parameters, self.grad_norm))
        assert np.isfinite(self.last_grad_norm) and self.last_grad_norm > 0
        if self.dry_run:
            return
        if self.approval is None:
            raise RuntimeError('Training requires the user-reviewed launch approval.')
        self.approval()
        self.optimizer.step()
        self.optimizer_steps += 1


def make_ppo(cfg, saved, device, output, num_envs=16, horizon=16, minibatch=256,
             learning_rate=1e-4, dry_run=True, approval=None):
    params = copy.deepcopy(cfg['train']['params'])
    wm_cfg = cfg['task']['env']['cameraPolicy']['worldModel']
    params['network'].update(world_model_enabled=True, wm_latent_dim=wm_cfg['wm_latent_dim'],
                             wm_feature_hidden=wm_cfg['wm_feature_hidden'])
    config = params['config']
    config.pop('central_value_config', None)
    config.update(device=str(device), device_name=str(device), env_name='unionmimic-ppo',
                  prefix='ppo', train_dir=str(Path(output).resolve()),
                  features={'observer': AlgoObserver()}, multi_gpu=False, mixed_precision=False,
                  num_actors=num_envs, horizon_length=horizon, minibatch_size=minibatch,
                  learning_rate=learning_rate, world_model_enabled=True,
                  reward_shaper=DefaultRewardsShaper(scale_value=.01),
                  env_info=dict(action_space=gym.spaces.Box(-1., 1., (22,), dtype=np.float32),
                                observation_space=gym.spaces.Dict({
                                    'obs':gym.spaces.Box(-np.inf,np.inf,(292,),dtype=np.float32),
                                    'wm_feature':gym.spaces.Box(-np.inf,np.inf,(512,),dtype=np.float32)}),
                                agents=1, value_size=1))
    agent = AdaptedPPO(params, dry_run=dry_run, approval=approval)
    # This adapter calls the tensor Isaac Gym task directly, without the
    # upstream rollout's first-observation inference of this flag.
    agent.is_tensor_obses = True
    agent.model.load_state_dict(saved['model'], strict=True)
    agent.model.eval()
    assert len(agent.optimizer.state) == 0, 'Warm start must use a fresh optimizer'
    return agent


def policy_command(agent, actions):
    """Reuse upstream clipping/rescaling BEFORE environment action noise.

    Keep the unmodified Gaussian sample for its PPO log probability; the
    simulator receives the transformed command, exactly as A2CBase.env_step.
    """
    return agent.preprocess_actions(actions)


def policy_observation(obs):
    return {'obs':obs['obs'].detach().clamp(-5,5).clone(),
            'wm_feature':obs['wm_feature'].detach().clone()}


@torch.no_grad()
def sample_policy(agent, obs):
    agent.model.eval()
    return agent.model({'obs': policy_observation(obs), 'is_train':False,
                        'prev_actions':None, 'rnn_states':None})


@torch.no_grad()
def live_critic_input(env):
    return critic_input(env.wm_adapter.world_model, env.wm_adapter._wm_latent,
                        env.obs_buf, env.prev_targets, env.last_actions).detach().clone()


def ppo_minibatch(agent, batch, ids):
    advantage = batch['advantage'][ids]
    return dict(old_values=batch['value'][ids,None], old_logp_actions=batch['logp'][ids],
                advantages=advantage, mu=batch['mu'][ids], sigma=batch['sigma'][ids],
                returns=batch['return'][ids,None], actions=batch['action'][ids],
                obs={'obs':batch['history'][ids], 'wm_feature':batch['deter'][ids]})


@torch.no_grad()
def collect_ppo_block(env, player, agent, critic, obs, horizon=16):
    from .collect import terminal_flags
    rows = []
    for _ in range(horizon):
        invalid = env.reset_buf.bool().clone()
        po = policy_observation(obs)
        ci = live_critic_input(env)
        result = sample_policy(agent, obs)
        # A simulator reset occurs in pre_physics_step. This incoming-reset
        # row is not a transition out of the previous terminal observation.
        obs, _, _, _ = player.env_step(env, policy_command(agent,result['actions']))
        flags = terminal_flags(env)
        next_ci = live_critic_input(env)
        rows.append(dict(history=po['obs'], deter=po['wm_feature'], critic_input=ci,
                         action=result['actions'].clone(), mu=result['mus'].clone(),
                         sigma=result['sigmas'].clone(), logp=result['neglogpacs'].clone(),
                         value=critic(ci).squeeze(-1), next_value=critic(next_ci).squeeze(-1),
                         reward=env.rew_buf.detach().clone()*.01,
                         failure=flags['failure'].clone(), timeout=flags['timeout'].clone(),
                         valid=~invalid))
    batch = {k:torch.stack([row[k] for row in rows]) for k in rows[0]}
    batch['advantage'], batch['return'] = generalized_advantage(
        batch['reward'], batch['value'], batch['next_value'], batch['failure'],
        batch['timeout'], batch['valid'], agent.gamma, agent.tau)
    valid = batch['valid'].flatten()
    flat = {k:v.flatten(0,1)[valid] for k,v in batch.items()}
    return flat, obs, int((~valid).sum())


def update_ppo_epoch(agent, critic, critic_opt, data, minibatch=4096, mini_epochs=4,
                     deadline=float('inf'), approval=None, dry_run=True):
    """Same loop in preflight and pilot; dry-run forbids both optimizers."""
    assert agent.dry_run == dry_run
    n=len(data['reward']);agent.set_train();kls=[];critic_steps=0;backward_batches=0
    data['advantage']=(data['advantage']-data['advantage'].mean())/(data['advantage'].std(unbiased=False)+1e-8)
    for _ in range(mini_epochs):
        mini_kl=[]
        for ids in torch.randperm(n,device=data['reward'].device).split(minibatch):
            if time.monotonic()>=deadline:break
            agent.calc_gradients(ppo_minibatch(agent,data,ids))
            mini_kl.append(float(agent.train_result[3]))
            # Same adaptive-KL references as upstream; old log-prob is fixed.
            data['mu'][ids]=agent.train_result[6];data['sigma'][ids]=agent.train_result[7]
            critic_opt.zero_grad(set_to_none=True)
            loss=(critic(data['critic_input'][ids]).squeeze(-1)-data['return'][ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO critic loss')
            loss.backward();norm=nn.utils.clip_grad_norm_(critic.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite PPO critic gradient')
            backward_batches+=1
            if not dry_run:
                if approval is None:raise PermissionError('Critic update requires review')
                approval();critic_opt.step();critic_steps+=1
        if mini_kl:
            kls.extend(mini_kl)
            agent.last_lr,agent.entropy_coef=agent.scheduler.update(
                agent.last_lr,agent.entropy_coef,agent.epoch_num,0,sum(mini_kl)/len(mini_kl))
            agent.update_lr(agent.last_lr)
        if time.monotonic()>=deadline:break
    if not backward_batches:return None
    return dict(actor_loss=float(agent.train_result[0]),critic_loss=float(loss),
                actor_grad_norm=agent.last_grad_norm,mean_kl=sum(kls)/len(kls),
                actor_lr=agent.last_lr,backward_batches=backward_batches,critic_steps=critic_steps)


def pilot_checkpoint(saved, agent, critic, critic_opt, epoch, frames, manifest_sha256, report):
    bundle={**saved,'model':agent.model.state_dict(),'ppo_state':dict(
        critic=critic.state_dict(),actor_optimizer=agent.optimizer.state_dict(),
        critic_optimizer=critic_opt.state_dict(),epoch=epoch,physical_frames=frames,
        actor_lr=agent.last_lr,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(agent.ppo_device),manifest_sha256=manifest_sha256),
        'ppo_report':report}
    for key in ('sm_state','sm_report'):bundle.pop(key,None)
    return bundle
