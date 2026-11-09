from .runtime import make_runtime, close_runtime
import json
import time
from pathlib import Path
import torch
from .collect import terminal_flags
from .common import write_json, file_digest


def evaluate_policy(args):
    output=Path(args.output).resolve()
    env,player,cfg,obs=make_runtime(args.envs,args.gpu,args.seed,args.checkpoint)
    try:
        live=torch.ones(args.envs,dtype=torch.bool,device=env.device)
        returns=torch.zeros(args.envs,device=env.device)
        lengths=torch.zeros(args.envs,dtype=torch.long,device=env.device)
        failures=torch.zeros_like(live)
        falls=torch.zeros_like(live)
        timeouts=torch.zeros_like(live)
        for step in range(env.max_episode_length+2):
            action=player.get_action(obs,True)
            obs,_,_,_=player.env_step(env,action)
            returns+=env.rew_buf*live
            lengths+=live.long()
            flags=terminal_flags(env)
            ending=live & env.reset_buf.bool()
            failures |= ending & flags['failure']
            falls |= ending & flags['fall']
            timeouts |= ending & flags['timeout']
            live &= ~ending
            if (step+1)%100==0:
                print(json.dumps(dict(phase='evaluate',steps=step+1,
                                      active_episodes=int(live.sum()),
                                      accumulated_mean_return=float(returns.mean()))),flush=True)
            if not bool(live.any()):
                break
        assert not bool(live.any())
        result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=file_digest(args.checkpoint),
                    seed=args.seed,episodes=args.envs,mean_return=float(returns.mean()),
                    returns=returns.cpu().tolist(),lengths=lengths.cpu().tolist(),
                    failures=int(failures.sum()),falls=int(falls.sum()),timeouts=int(timeouts.sum()),
                    failure_rate=float(failures.float().mean()))
        write_json(output,result)
        print(json.dumps(result),flush=True)
    finally:
        close_runtime(env)
