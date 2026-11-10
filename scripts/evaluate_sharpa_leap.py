#!/usr/bin/env python3
"""Independent first-episode Sharpa evaluation; all envs count, including early failures."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym  # noqa: F401
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import argparse,json,time,traceback
import torch
from in_hand_rotation_mjlab.world_model.policy import Actor
from unionmimic.common import write_json,file_digest
from unionmimic.online_baseline.simulator import make_simulator,state,context


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=20261105)
    a=ap.parse_args();out=Path(a.out).resolve()
    if out.exists():raise FileExistsError(out)
    started=time.time();torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu');actor=Actor.from_checkpoint(ck['actor']).to('cuda:%d'%a.gpu).eval()
    assert actor.obs_dim==730 and actor.act_dim==22 and actor.hidden==(512,512,256)
    spin_sign=ck.get('spin_sign',1)
    env,task,_=make_simulator(512,a.gpu,a.seed,spin_sign=spin_sign);device=env.device
    assert bool((env.spin_axis[:,2]==spin_sign).all())
    frame=env.obs_buf[:,:73].clamp(-5,5);history=frame[:,None].expand(-1,10,-1).clone()
    active=torch.ones(512,dtype=torch.bool,device=device);returns=torch.zeros(512,device=device)
    rotations=torch.zeros_like(returns);offaxis=torch.zeros_like(returns);tilts=torch.zeros_like(returns);steps=torch.zeros_like(returns)
    failed=torch.zeros_like(active);timed=torch.zeros_like(active);reward_error=0.
    with torch.no_grad():
        for t in range(600):
            if time.time()-started>570:raise TimeoutError('Evaluation deadline')
            invalid=env.reset_buf.bool().clone();valid=active&~invalid
            s=state(env);c=context(env);env.step(actor(history.flatten(1)))
            sn=state(env);r,f,timeout,_,metrics=task.reward(s,sn,c)
            if valid.any():reward_error=max(reward_error,float((r[valid]-env.rew_buf[valid]).abs().max()))
            for tensor in (env.rew_buf,env.obs_buf,r):assert torch.isfinite(tensor).all(),'Nonfinite evaluation'
            returns[valid]+=env.rew_buf[valid];rotations[valid]+=metrics['rotation'][valid]
            offaxis[valid]+=metrics['offaxis'][valid];tilts[valid]+=metrics['tilt'][valid];steps[valid]+=1
            done=env.reset_buf.bool()&valid
            failed[done]=f[done];timed[done]=timeout[done];active[done]=False
            frame=env.obs_buf[:,:73].clamp(-5,5);history=torch.cat((history[:,1:],frame[:,None]),1)
            history[invalid]=frame[invalid,None].expand(-1,10,-1)
            if not active.any():break
    assert not active.any(),'Incomplete first episodes';assert reward_error<=2e-3
    rows=[dict(return_raw=float(returns[i]),rotation_rad=float(rotations[i]),offaxis_rad=float(offaxis[i]),
               axis_deviation_rad=float(tilts[i]/steps[i].clamp_min(1)),steps=int(steps[i]),failure=bool(failed[i]),timeout=bool(timed[i])) for i in range(512)]
    result=dict(passed=True,episodes=512,seed=a.seed,protocol='First completed valid policy episode per env, no survivor filtering',
        mean_return=float(returns.mean()),mean_rotation_rad=float(rotations.mean()),failure_rate=float(failed.float().mean()),
        mean_axis_deviation_rad=float((tilts/steps.clamp_min(1)).mean()),reward_max_error=reward_error,
        seconds=time.time()-started,checkpoint_sha256=file_digest(a.checkpoint),spin_sign=spin_sign,rows=rows)
    write_json(out,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
