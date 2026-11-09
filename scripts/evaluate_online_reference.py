#!/usr/bin/env python
"""Frozen released-policy ability reference under the long study's dev protocol."""
import os,sys,traceback,time,json,argparse
from pathlib import Path
from unionmimic.runtime import make_runtime
import torch
import numpy as np
from unionmimic.common import RELEASE,write_json,file_digest,digest
from unionmimic.collect import terminal_flags
from unionmimic.online.task import qrotate,qconj

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.time()
    write_json(out/'status.json',dict(status='running',gpu=a.gpu,pid=os.getpid(),started_unix=start))
    try:
        env,player,cfg,obs=make_runtime(16,a.gpu,a.seed,RELEASE,overrides=['test=False'])
        before=digest(player.model)
        obs,_,_,_=player.env_step(env,torch.zeros(16,22,device=env.device))
        sums={k:torch.zeros(16,device=env.device) for k in ['return','rotation','offaxis','tilt_sum','steps']}
        first={}
        with torch.no_grad():
            while len(first)<16:
                if time.time()-start>900:raise TimeoutError('Reference evaluation budget exceeded')
                command=player.get_action(obs,is_deterministic=True)
                obs,_,_,_=player.env_step(env,command)
                flags=terminal_flags(env)
                axis=qrotate(env.object_rot,qrotate(qconj(env.object_init_quat),env.spin_axis))
                tilt=torch.acos((axis*env.spin_axis).sum(-1).clamp(-1,1))
                sums['return']+=env.rew_buf;sums['rotation']+=env.last_spin_delta_axis/20
                sums['offaxis']+=env.last_spin_delta_offaxis/20;sums['tilt_sum']+=tilt;sums['steps']+=1
                for i in env.reset_buf.nonzero().flatten().tolist():
                    if i in first:continue
                    row={k:float(v[i]) for k,v in sums.items()}
                    first[i]=dict(row,env_id=i,seed=a.seed,failure=bool(flags['failure'][i]),
                                  timeout=bool(flags['timeout'][i]),axis_deviation_rad=row['tilt_sum']/row['steps'])
        assert before==digest(player.model)
        episodes=[first[i] for i in range(16)]
        write_json(out/'evaluation.json',dict(checkpoint=str(RELEASE),checkpoint_sha256=file_digest(RELEASE),
            episodes=episodes,mean_return=float(np.mean([e['return'] for e in episodes])),
            mean_rotation=float(np.mean([e['rotation'] for e in episodes])),failures=sum(e['failure'] for e in episodes),
            parameters_unchanged=True,role='Evaluation-only ability reference; never provides training actions or replay.',
            seconds=time.time()-start))
        write_json(out/'status.json',dict(status='completed',gpu=a.gpu,pid=os.getpid(),started_unix=start,ended_unix=time.time()))
    except BaseException as e:
        write_json(out/'status.json',dict(status='failed',gpu=a.gpu,pid=os.getpid(),started_unix=start,ended_unix=time.time(),error=repr(e)));raise

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    finally:
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        sys.stdout.flush();sys.stderr.flush()
    os._exit(code)
