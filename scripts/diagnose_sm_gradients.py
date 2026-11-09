"""Measure reward/bootstrap/anchor gradients without updating any parameters."""
import argparse
import copy
from pathlib import Path
import unionmimic.runtime  # Isaac Gym import must precede torch.
import torch
from torch import nn
from unionmimic.models import load_world_model, CalibratedHeads, state_from_cache
from unionmimic.sm import make_player, DifferentiableActor, control_map, critic_input
from unionmimic.common import write_json, file_digest


def main(args):
    cache=torch.load(str(args.cache),map_location='cpu');meta=cache['metadata'];train=cache['train']
    wm,saved,cfg=load_world_model(Path(meta['dataset'])/'config.yaml',args.checkpoint,args.gpu)
    device=next(wm.parameters()).device
    heads=CalibratedHeads(wm).to(device).eval();heads.failure.load_state_dict(saved['failure_head'])
    heads.failure_log_weight.fill_(saved['failure_log_weight']);heads.requires_grad_(False)
    actor=DifferentiableActor(make_player(cfg,saved,device).model).to(device).eval()
    anchor=copy.deepcopy(actor).requires_grad_(False);anchor.model.load_state_dict(saved['sm_anchor'])
    critic=nn.Sequential(nn.Linear(wm._feat_size+292+44,256),nn.LayerNorm(256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,1)).to(device)
    critic.load_state_dict(saved['sm_state']['critic']);critic.requires_grad_(False)
    eligible=torch.nonzero(~train['stress']).flatten()
    lower=torch.tensor(meta['lower_limits'],device=device);upper=torch.tensor(meta['upper_limits'],device=device)
    params=[p for p in actor.parameters() if p.requires_grad]
    def gradient(value):
        grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])
    reports=[]
    for seed in (711,712,713):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        ids=eligible[torch.randint(len(eligible),(args.batch,))]
        state=state_from_cache(train,ids,device);history=train['start_history'][ids].to(device)
        targets=train['start_targets'][ids].to(device);last=train['start_last_actions'][ids].to(device)
        bias=torch.randn(args.batch,22,device=device)*.015
        survival=torch.ones(args.batch,device=device);reward_sum=torch.zeros_like(survival);anchor_loss=0.
        saturation=[]
        for h in range(4):
            command=actor(history,state['deter']);reference=anchor(history,state['deter']).detach()
            saturation.append(float((command.abs()>=.999).float().mean()))
            anchor_loss=anchor_loss+(command-reference).square().mean()/4
            noisy=(command+torch.randn_like(command)*.04+bias).clamp(-1,1)
            scale=.2*(1+(torch.rand(args.batch,1,device=device)-.5)*.1)
            action,targets=control_map(noisy,last,targets,scale,lower,upper)
            state=wm.dynamics.img_step(state,action);feat=wm.dynamics.get_feat(state)
            rd,pd,_=heads.predictions(feat)
            reward_sum=reward_sum+.99**h*survival*rd.mode().reshape(-1)*.01
            survival=survival*(1-heads.failure_probability(feat))
            frame=torch.cat([pd.mode()[:,:22],2*(targets-lower)/(upper-lower)-1,pd.mode()[:,44:49].clamp(0,1),history[:,49:73]],-1)
            history=torch.cat([frame,history[:,:219]],-1);last=action
        tail=.99**4*survival*critic(critic_input(wm,state,history,targets,last)).squeeze(-1)
        r=gradient(reward_sum.mean());v=gradient(tail.mean());a=gradient(args.anchor*anchor_loss)
        full=-r-v+a
        def cosine(x,y):return float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-12))
        reports.append(dict(seed=seed,reward_4step=float(reward_sum.mean()),bootstrap=float(tail.mean()),
                            weighted_anchor_loss=float(args.anchor*anchor_loss),reward_gradient_norm=float(r.norm()),
                            bootstrap_gradient_norm=float(v.norm()),anchor_gradient_norm=float(a.norm()),
                            total_gradient_norm=float(full.norm()),cosine_reward_bootstrap=cosine(r,v),
                            cosine_reward_total_descent=cosine(r,-full),
                            saturation_fraction=sum(saturation)/4,
                            bootstrap_share_of_objective=float(tail.abs().mean()/(tail.abs().mean()+reward_sum.abs().mean())),
                            history_policy_obs_rmse=float((train['start_history'][ids]-train['start_policy_obs'][ids]).square().mean().sqrt())))
    write_json(args.output,dict(checkpoint=str(args.checkpoint),sha256=file_digest(args.checkpoint),cache=str(args.cache),
                               batch=args.batch,anchor_coef=args.anchor,sm_step=saved['sm_state']['step'],measurements=reports))
    print(reports,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--anchor',type=float,default=10.)
    main(p.parse_args())
