"""Four-step actor BPTT through the calibrated WM-Craftnet RSSM."""
import copy
import json
import time
from pathlib import Path
import gym
import numpy as np
import torch
from torch import nn
from rl_games.algos_torch.players import PpoPlayerContinuous
from .common import digest, save_checkpoint, write_json
from .models import load_world_model, CalibratedHeads, state_model_digest, state_from_cache


def make_player(cfg, saved, device):
    params=copy.deepcopy(cfg['train']['params'])
    wm_cfg=cfg['task']['env']['cameraPolicy']['worldModel']
    params['network'].update(world_model_enabled=True,wm_latent_dim=wm_cfg['wm_latent_dim'],
                             wm_feature_hidden=wm_cfg['wm_feature_hidden'])
    params['config'].update(device_name=str(device),env_info=dict(
        action_space=gym.spaces.Box(-1.,1.,shape=(22,),dtype=np.float32),
        observation_space=gym.spaces.Dict({
            'obs':gym.spaces.Box(-np.inf,np.inf,shape=(292,),dtype=np.float32),
            'wm_feature':gym.spaces.Box(-np.inf,np.inf,shape=(512,),dtype=np.float32),
            'depth':gym.spaces.Box(0.,1.,shape=(72,96,1),dtype=np.float32)}),
        agents=1,value_size=1))
    player=PpoPlayerContinuous(params)
    player.model.load_state_dict(saved['model'],strict=True)
    player.model.eval()
    player.has_batch_dimension=True
    player.is_tensor_obses=True
    player.init_rnn()
    assert not player.is_rnn, 'This adapter handles the published feed-forward actor'
    return player


class DifferentiableActor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model

    def forward(self, history, deter):
        # Calling the frozen-statistics normalizer directly retains gradients
        # to predicted observations; BaseModelNetwork.norm_obs uses no_grad.
        history=history.clamp(-5,5)
        normalized=self.model.running_mean_std(history)
        result=self.model.a2c_network({'obs':{'obs':normalized,'wm_feature':deter},
                                       'rnn_states':None})
        return result[0].clamp(-1,1)


def control_map(command, previous_action, targets, scale, lower, upper):
    action=.8*command.clamp(-1,1)+.2*previous_action
    targets=torch.maximum(torch.minimum(targets+scale*action,upper),lower)
    return action,targets


def critic_input(wm,state,history,targets,last_action):
    return torch.cat([wm.dynamics.get_feat(state),history,targets,last_action],-1)


def lambda_returns(rewards, continuation, values, gamma=.99, lam=.95):
    result=[]
    ret=values[-1]
    for t in reversed(range(len(rewards))):
        ret=rewards[t]+gamma*continuation[t]*((1-lam)*values[t+1]+lam*ret)
        result.append(ret)
    return torch.stack(result[::-1])


def imagined_return(rewards, continuation, tail_value, gamma=.99):
    """Survival-weighted reward plus terminal bootstrap (SM Eq. 2)."""
    survival=torch.ones_like(tail_value)
    total=torch.zeros_like(tail_value)
    for h,(reward,cont) in enumerate(zip(rewards,continuation)):
        total=total+(gamma**h)*survival*reward
        survival=survival*cont
    return total+(gamma**len(rewards))*survival*tail_value


def train_sm(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    cache=torch.load(str(args.cache),map_location='cpu')
    meta,train=cache['metadata'],cache['train']
    torch.manual_seed(args.seed)
    wm,saved,cfg=load_world_model(Path(meta['dataset'])/'config.yaml',args.checkpoint,args.gpu)
    assert saved['calibration_report']['passed'], 'Prediction calibration gate has not passed'
    assert state_model_digest(wm)==meta['state_model_sha256'], 'Recompute starts after a dynamics update'
    device=next(wm.parameters()).device
    heads=CalibratedHeads(wm).to(device)
    heads.failure.load_state_dict(saved['failure_head'])
    heads.failure_log_weight.fill_(saved['failure_log_weight'])
    heads.requires_grad_(False)
    heads.eval()
    player=make_player(cfg,saved,device)
    actor=DifferentiableActor(player.model).to(device).eval()
    anchor=copy.deepcopy(actor).requires_grad_(False)
    if 'sm_anchor' in saved:
        anchor.model.load_state_dict(saved['sm_anchor'])
    critic=nn.Sequential(nn.Linear(wm._feat_size+292+44,256),nn.LayerNorm(256),nn.ELU(),
                         nn.Linear(256,256),nn.ELU(),nn.Linear(256,1)).to(device)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=args.lr)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=3e-4)
    start_step=0
    if args.resume:
        sm=saved['sm_state']
        critic.load_state_dict(sm['critic'])
        actor_opt.load_state_dict(sm['actor_optimizer'])
        critic_opt.load_state_dict(sm['critic_optimizer'])
        start_step=sm['step']
        # Explicit run configuration takes precedence over the saved LR.
        for group in actor_opt.param_groups:
            group['lr']=args.lr
    lower=torch.tensor(meta['lower_limits'],device=device)
    upper=torch.tensor(meta['upper_limits'],device=device)
    eligible=torch.nonzero(~train['stress'],as_tuple=False).flatten()
    assert len(eligible)>args.batch
    ids=eligible[:args.batch]
    history=train['start_policy_obs'][ids].to(device).requires_grad_(True)
    deter=train['state_deter'][ids].to(device).requires_grad_(True)
    before_norm=digest(actor.model.running_mean_std)
    with torch.no_grad():
        expected=player.get_action({'obs':history.detach().clone(),'wm_feature':deter.detach().clone()},True)
    actual=actor(history,deter)
    parity=float((actual-expected).abs().max())
    assert parity<1e-6, 'Differentiable actor changes checkpoint behavior'
    gradients=torch.autograd.grad(actual.square().mean(),[history,deter])
    assert all(bool(torch.isfinite(g).all()) and float(g.abs().sum())>0 for g in gradients)
    wm_before=digest(wm)
    failure_before=digest(heads.failure)
    actor_before=digest(actor)
    if args.resume:
        # Parity validation invokes the original stochastic player, even when
        # selecting its mean. Restore RNG only after that validation.
        torch.set_rng_state(sm['torch_rng'])
        torch.cuda.set_rng_state(sm['cuda_rng'],device)
    if args.fork_seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    began=time.monotonic()
    horizon=meta['horizon']
    with open(out/'metrics.jsonl','a',buffering=1) as log:
        for step in range(start_step+1,start_step+args.updates+1):
            ids=eligible[torch.randint(len(eligible),(args.batch,))]
            state=state_from_cache(train,ids,device)
            history=train['start_history'][ids].to(device)
            targets=train['start_targets'][ids].to(device)
            last_action=train['start_last_actions'][ids].to(device)
            action_bias=torch.randn(args.batch,22,device=device)*.015
            rows,rewards,continuations=[],[],[]
            anchor_loss=0.
            critic_opt.zero_grad(set_to_none=True)
            critic.requires_grad_(False)
            critic_before=digest(critic) if step==start_step+1 else None
            for h in range(horizon):
                rows.append(critic_input(wm,state,history,targets,last_action).detach())
                command=actor(history,state['deter'])
                with torch.no_grad():
                    reference=anchor(history,state['deter'])
                anchor_loss=anchor_loss+(command-reference).square().mean()/horizon
                noisy=(command+torch.randn_like(command)*.04+action_bias).clamp(-1,1)
                # Native target scale is one random scalar per environment.
                scale=.2*(1+(torch.rand(args.batch,1,device=device)-.5)*.1)
                action,targets=control_map(noisy,last_action,targets,scale,lower,upper)
                state=wm.dynamics.img_step(state,action)  # straight-through stochastic categorical state
                feat=wm.dynamics.get_feat(state)
                rd,pd,_=heads.predictions(feat)
                reward=rd.mode().reshape(-1)*.01  # match PPO reward scaling
                continuation=1-heads.failure_probability(feat)
                prop=pd.mode()
                normalized_targets=2*(targets-lower)/(upper-lower)-1
                frame=torch.cat([prop[:,:22],normalized_targets,prop[:,44:49].clamp(0,1),history[:,49:73]],-1)
                history=torch.cat([frame,history[:,:219]],-1)
                last_action=action
                rewards.append(reward)
                continuations.append(continuation)
            end=critic_input(wm,state,history,targets,last_action)
            objective=imagined_return(rewards,continuations,critic(end).squeeze(-1),args.gamma)
            loss=-objective.mean()+args.anchor_coef*anchor_loss
            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
            assert bool(torch.isfinite(grad)) and float(grad)>0
            assert all(p.grad is None for p in heads.parameters()), 'Actor update wrote model gradients'
            actor_opt.step()
            if critic_before is not None:
                assert digest(critic)==critic_before, 'Actor update changed critic parameters'
            rows.append(end.detach())
            rows=torch.stack(rows)
            with torch.no_grad():
                values=critic(rows).squeeze(-1)
                target=lambda_returns(rewards,continuations,values,args.gamma,.95)
            critic.requires_grad_(True)
            value_loss=(critic(rows[:-1]).squeeze(-1)-target).square().mean()
            critic_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            value_grad=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.)
            assert bool(torch.isfinite(value_grad)) and bool(torch.isfinite(value_loss))
            critic_opt.step()
            if step%25==0 or step==start_step+1:
                row=dict(phase='sm',step=step,actor_loss=float(loss),critic_loss=float(value_loss),
                         actor_grad_norm=float(grad),imagined_return=float(objective.mean()),
                         anchor_mse=float(anchor_loss),seconds=round(time.monotonic()-began,1))
                print(json.dumps(row),flush=True)
                log.write(json.dumps(row)+'\n')
            if step%args.save_every and step!=start_step+args.updates:
                continue
            assert digest(wm)==wm_before and digest(heads.failure)==failure_before
            assert digest(actor.model.running_mean_std)==before_norm
            report=dict(step=step,actor_parity_max_error=parity,
                        input_gradient_norms=[float(g.norm()) for g in gradients],
                        actor_changed=digest(actor)!=actor_before,world_model_unchanged=True,
                        failure_head_unchanged=True,normalization_unchanged=True,
                        critic_frozen_during_actor_update=True,horizon=horizon,
                        reward_scale=.01,seconds=time.monotonic()-began,
                        training_mode='short_horizon_imagination_from_real_episode_starts')
            report.update(actor_lr=args.lr,anchor_coefficient=args.anchor_coef,seed=args.seed,
                          fork_seed=args.fork_seed)
            bundle={**saved,'model':actor.model.state_dict(),'sm_anchor':anchor.model.state_dict(),
                    'sm_state':dict(step=step,critic=critic.state_dict(),
                        actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)),
                    'sm_report':report}
            save_checkpoint(out/'last.pt',bundle)
            write_json(out/'status.json',report)
        write_json(out/'completed.json',report)
