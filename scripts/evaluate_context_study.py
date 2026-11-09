"""Independent first-episode evaluation with frozen real WSM state and actor."""
import ctypes, os, sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import argparse,json,time,traceback
import torch
from unionmimic.context_study import make_context_env,ContextActor
from unionmimic.online_baseline.simulator import state,context
from unionmimic.common import write_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261213)
    p.add_argument('--pulse',action='store_true');p.add_argument('--envs',type=int,default=128)
    a=p.parse_args();ckpath=Path(a.checkpoint).resolve();out=Path(a.out).resolve();assert not out.exists()
    torch.set_num_threads(4);started=time.time();ck=torch.load(ckpath,map_location='cpu')
    env,task,_=make_context_env(a.envs,a.gpu,a.seed,ROOT/'work/analysis/sm-context-trajectories-20261109-v1/config.yaml')
    actor=ContextActor.from_checkpoint(ck['actor']).to(env.device).eval().requires_grad_(False)
    h=env.obs_buf[:,:73].clamp(-5,5)[:,None].expand(-1,10,-1).clone()
    active=torch.ones(a.envs,dtype=torch.bool,device=env.device)
    vals={k:torch.zeros(a.envs,device=env.device) for k in ('return','rotation','offaxis','tilt','steps','failed','initial_reset_failure')}
    traces=[];error=0.
    noisegen=torch.Generator(device=env.device).manual_seed(a.seed+100)
    with torch.no_grad():
        for t in range(600):
            assert time.time()-started<600,'Evaluation timeout'
            invalid=env.reset_buf.bool().clone();valid=active&~invalid
            s=state(env);c=context(env);actor.real_context=env.wm_adapter._wm_latent['deter']
            action=actor(h.flatten(1))
            if a.pulse and 40<=t<50: action=(action+.3*torch.randn(action.shape,device=env.device,generator=noisegen)).clamp(-1,1)
            env.step(action);sn=state(env);r,f,timeout,_,m=task.reward(s,sn,c)
            if valid.any():error=max(error,float((r[valid]-env.rew_buf[valid]).abs().max()))
            assert torch.isfinite(r).all() and error<=2e-3
            for key,source in [('return',env.rew_buf),('rotation',m['rotation']),('offaxis',m['offaxis']),('tilt',m['tilt'])]:vals[key][valid]+=source[valid]
            vals['steps'][valid]+=1
            done=env.reset_buf.bool()&valid;vals['failed'][done]=f[done].float();active[done]=False
            # Count, rather than silently replace, an environment that fails in its reset frame.
            reset_failed=active&invalid&env.reset_buf.bool();vals['initial_reset_failure'][reset_failed]=1
            vals['failed'][reset_failed]=1;active[reset_failed]=False
            traces.append(dict(step=t,active=int(active.sum()),rotation_per_s=float(m['rotation'][valid].mean()/.10002) if valid.any() else 0.))
            frame=env.obs_buf[:,:73].clamp(-5,5);h=torch.cat((h[:,1:],frame[:,None]),1);h[invalid]=frame[invalid,None].expand(-1,10,-1)
            if not active.any():break
    assert not active.any(),'Incomplete episodes'
    result=dict(passed=True,mode=actor.mode,episodes=a.envs,seed=a.seed,pulse=a.pulse,checkpoint_sha256=file_digest(ckpath),
        protocol='Every initial environment, first valid episode; reset-frame failure counts. Pulse is command noise .3 at steps40:50, not force.',
        mean_return=float(vals['return'].mean()),mean_rotation_rad=float(vals['rotation'].mean()),failure_rate=float(vals['failed'].mean()),
        mean_axis_deviation_rad=float((vals['tilt']/vals['steps'].clamp_min(1)).mean()),mean_offaxis_rad=float(vals['offaxis'].mean()),
        mean_steps=float(vals['steps'].mean()),reward_max_error=error,seconds=time.time()-started,
        rows=[{k:float(v[i]) for k,v in vals.items()} for i in range(a.envs)],trace=traces)
    write_json(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','trace')}),flush=True)
if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
