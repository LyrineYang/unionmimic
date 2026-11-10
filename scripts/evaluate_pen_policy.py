"""Fixed first-frame evaluation shared by SM/PPO; reports tracking, not just return."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import argparse,time,json,traceback
import torch
from in_hand_rotation_mjlab.world_model.policy import Actor
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.pen_tasks import TASKS
from unionmimic.common import file_digest,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--retarget',required=True);p.add_argument('--task-set',choices=['all','reference'],default='all');p.add_argument('--envs',type=int,default=512)
    p.add_argument('--single-task',type=int,choices=(2,3));p.add_argument('--single-clip',type=int)
    p.add_argument('--seed',type=int,default=20261199)
    p.add_argument('--zero-residual',action='store_true')
    a=p.parse_args();out=Path(a.out).resolve();checkpoint=Path(a.checkpoint).resolve()
    ckpt=torch.load(checkpoint,map_location='cpu');retarget=Path(a.retarget).resolve();start=time.time();torch.set_num_threads(4)
    assert (a.single_task is None)==(a.single_clip is None)
    if a.single_clip is not None:
        assert (ckpt['task'],ckpt['clip'])==(a.single_task,a.single_clip)
        assert ckpt['retarget_manifest_sha256']==file_digest(retarget/'manifest.json')
    env,_=make_pen_simulator(a.envs,a.gpu,a.seed,retarget,fixed_phase=0,
                             physics_numerics=ckpt.get('physics_numerics'));dev=env.device
    actor=Actor.from_checkpoint(ckpt['actor']).to(dev);actor.eval()
    tasks=[a.single_task] if a.single_task is not None else ([0,1,2,3] if a.task_set=='all' else [2,3])
    env.task_ids[:]=torch.tensor(tasks,device=dev)[torch.arange(a.envs,device=dev)%len(tasks)]
    env.fixed_clips=True
    for task in tasks:
        ids=(env.task_ids==task).nonzero().flatten();pool=env.pen_tasks.multi if task==3 else env.pen_tasks.single
        if a.single_clip is not None:
            assert bool((pool==a.single_clip).any())
            env.clip_ids[ids]=a.single_clip
        else:
            env.clip_ids[ids]=pool[torch.arange(len(ids),device=dev)%len(pool)]
    env.fixed_phase=torch.where(env.task_ids>=2,-torch.ones_like(env.task_ids)*50,torch.zeros_like(env.task_ids))
    zero=torch.zeros(a.envs,22,device=dev)
    env.reset_buf.fill_(1);env.step(zero)
    history=env.policy_frame[:,None].expand(-1,10,-1).clone();done=env.pen_failure.clone();failed=done.clone()
    totals=torch.zeros(a.envs,4,device=dev);steps=torch.zeros(a.envs,device=dev);recovery=torch.full((a.envs,),-1.,device=dev);streak=torch.zeros_like(steps)
    peak=env.contact_tensor.view(a.envs,-1,3)[:,env.object_rb_handles].norm(dim=-1).amax(-1)
    maxsteps=max(int(env.pen_tasks.lengths.max())+51,301)
    with torch.no_grad():
        for i in range(maxsteps):
            if bool(done.all()):break
            active=~done;env.reset_buf.zero_();env.step(zero if a.zero_residual else actor(history.flatten(1)))
            m=env.pen_metrics;vals=torch.stack((env.rew_buf,m['position_error_m'],m['shaft_error_rad'],env.pen_failure.float()),-1)
            assert torch.isfinite(vals[active]).all()
            force=env.contact_tensor.view(a.envs,-1,3)[:,env.object_rb_handles].norm(dim=-1).amax(-1)
            assert torch.isfinite(force[active]).all()
            peak=torch.where(active,torch.maximum(peak,force),peak)
            totals[active]+=vals[active];steps[active]+=1
            recovered=(env.task_ids==1)&(env.reference_step>=110)&(m['position_error_m']<.01)&(m['shaft_error_rad']<torch.pi/18)
            streak=torch.where(recovered,streak+1,torch.zeros_like(streak))
            hit=active&(streak>=10)&(recovery<0);recovery[hit]=(env.reference_step[hit]-110-9).float()*.02
            failed|=active&env.pen_failure;done|=env.pen_failure|env.pen_timeout
            frame=env.policy_frame;history=torch.cat((history[:,1:],frame[:,None]),1)
    rows=[]
    for i in range(a.envs):
        rows.append(dict(task=int(env.task_ids[i]),clip=int(env.clip_ids[i]),return_=float(totals[i,0]),steps=int(steps[i]),
            position_error_mm=float(totals[i,1]/steps[i].clamp_min(1)*1000),shaft_error_deg=float(totals[i,2]/steps[i].clamp_min(1)*180/torch.pi),
            failure=bool(failed[i]),completed=bool(done[i]&~failed[i]),recovery_seconds=float(recovery[i]),
            peak_object_force_n=float(peak[i])))
    summaries={}
    for t in tasks:
        rs=[r for r in rows if r['task']==t];n=len(rs)
        summaries[TASKS[t]]=dict(episodes=n,mean_return=sum(r['return_'] for r in rs)/n,success_rate=sum(r['completed'] for r in rs)/n,
            failures=sum(r['failure'] for r in rs),completed_with_peak_at_most_50n=sum(r['completed'] and r['peak_object_force_n']<=50 for r in rs),
            mean_position_error_mm=sum(r['position_error_mm'] for r in rs)/n,
            mean_shaft_error_deg=sum(r['shaft_error_deg'] for r in rs)/n)
    write_json(out,dict(passed=True,episodes=a.envs,seed=a.seed,task_set=a.task_set,
        single_task=a.single_task,single_clip=a.single_clip,first_frame_hold_seconds=1,
        controller='zero_residual' if a.zero_residual else 'trained_actor',checkpoint_sha256=file_digest(checkpoint),
        summary=summaries,mean_return=float(totals[:,0].mean()),rows=rows,seconds=time.time()-start))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
