"""Historical SM controls; frozen WM encoder estimates geometry for display only.

Also records aligned physical transitions for a separate observation-bridge audit.
No policy update, no prior-to-action feedback, no selected successful episodes.
"""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from isaacgym import gymapi
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import argparse,json,time,traceback,subprocess
import numpy as np,torch,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from in_hand_rotation_mjlab.world_model.policy import Actor
from unionmimic.context_study import make_context_env
from unionmimic.geometry_context import PositionReadout
from unionmimic.online_baseline.simulator import state,context
from unionmimic.online_baseline.task import qrotate
from unionmimic.leap_sharpa.model import pack
from unionmimic.common import file_digest,write_json,digest,RELEASE,save_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=3);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261218);p.add_argument('--steps',type=int,default=300);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    actorpath=ROOT/'work/runs/sharpa-leap-12h-20261103-v1/ckpt/best.pt'
    headpath=ROOT/'work/analysis/geometry-readout-20261109-v1/pretrained.pt'
    ck=torch.load(actorpath,map_location='cpu');headck=torch.load(headpath,map_location='cpu')
    env,native,_=make_context_env(16,a.gpu,a.seed,ROOT/'work/analysis/sm-context-trajectories-20261109-v1/config.yaml');dev=env.device
    actor=Actor.from_checkpoint(ck['actor']).to(dev).eval().requires_grad_(False);head=PositionReadout.from_checkpoint(headck).to(dev).eval()
    wm=env.wm_adapter.world_model;assert digest(wm.encoder)==headck['encoder_sha256']
    hashes=dict(actor=digest(actor),prior=digest(wm),readout=digest(head))
    latest={};hook=wm.encoder.register_forward_hook(lambda m,x,y:latest.update(embed=y.detach()))
    with torch.no_grad():wm.encoder(dict(prop=env.last_obs_buf,image=env.obs_dict['obs']['depth']))
    frame=env.obs_buf[:,:73].clamp(-5,5);history=frame[:,None].expand(-1,10,-1).clone()
    props=gymapi.CameraProperties();props.width=960;props.height=720;props.horizontal_fov=45
    camera=env.gym.create_camera_sensor(env.envs[0],props)
    env.gym.set_camera_location(camera,env.envs[0],gymapi.Vec3(-.075,-.35,1.264),gymapi.Vec3(-.075,0,.964))
    view=np.array(env.gym.get_camera_view_matrix(env.sim,env.envs[0],camera));proj=np.array(env.gym.get_camera_proj_matrix(env.sim,env.envs[0],camera))
    def project(world):
        v=np.r_[world,1.]@view@proj
        return int((v[0]/v[3]+1)*480),int((1-v[1]/v[3])*360)
    font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1440x900','-r','10','-i','-','-an','-c:v','libx264','-crf','18','-threads','3','-pix_fmt','yuv420p',str(out/'geometry-demo.mp4')],stdin=subprocess.PIPE)
    data=[];display=[];frames=0;video_active=True;episode=torch.arange(16,device=dev);next_episode=16
    noise=torch.Generator(device=dev).manual_seed(a.seed+100);parity=0.
    try:
        with torch.no_grad():
            for t in range(a.steps):
                invalid=env.reset_buf.bool().clone();valid=~invalid
                if invalid.any():
                    ids=invalid.nonzero().flatten();episode[ids]=torch.arange(next_episode,next_episode+len(ids),device=dev);next_episode+=len(ids)
                base=env.root_state_tensor[env.hand_indices,:13].clone();s=pack(state(env),base);c=torch.cat((context(env),base[:,:7]),-1)
                estimated=head(history,latest['embed']);command=actor(history.flatten(1))
                pulse=120<=t<130
                if pulse:command=(command+.3*torch.randn(command.shape,device=dev,generator=noise)).clamp(-1,1)
                raw=state(env);env.step(command);sn=pack(state(env),base)
                r,failed,timeout,_,metrics=native.reward(raw,state(env),c[:,:93])
                if valid.any():parity=max(parity,float((r[valid]-env.rew_buf[valid]).abs().max()))
                assert parity<=2e-3
                frame=env.obs_buf[:,:73].clamp(-5,5);history=torch.cat((history[:,1:],frame[:,None]),1);history[invalid]=frame[invalid,None].expand(-1,10,-1)
                estimate_next=head(history,latest['embed'])
                data.append({k:v.detach().cpu().clone() for k,v in dict(state=s,next_state=sn,context=c,command=env.study_command,
                    scale=env.relative_scale_tensor,estimate=estimated,next_estimate=estimate_next,episode=torch.where(valid,episode,-1),done=env.reset_buf.bool()).items()})
                if not video_active:continue
                env.gym.fetch_results(env.sim,True);env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
                rgba=env.gym.get_camera_image(env.sim,env.envs[0],camera,gymapi.IMAGE_COLOR).reshape(720,960,4)
                image=Image.fromarray(rgba[:,:,:3]);draw=ImageDraw.Draw(image)
                truth=sn[0,66:69];pred=estimate_next[0];world_pred=qrotate(base[:1,3:7],pred[None])[0]+base[0,:3]
                pixel=project(world_pred.cpu().numpy());truthpixel=project(env.root_state_tensor[env.object_indices[0],:3].cpu().numpy())
                for point,color,radius in [(truthpixel,'#ffffff',8),(pixel,'#51e6bf',13)]:
                    x,y=point
                    if 0<=x<960 and 0<=y<720:draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=3)
                canvas=Image.new('RGB',(1440,900),'#0b1220');canvas.paste(image,(25,130));d=ImageDraw.Draw(canvas)
                d.text((30,24),'UNIONMIMIC / Seeing the object inside the hand',font=font(29),fill='#edf5ff')
                d.text((30,71),'Historical SM controls. Frozen WM-Craftnet encoder observes.',font=font(22),fill='#aac0d4')
                d.text((1010,152),'LIVE GEOMETRY',font=font(23),fill='#51e6bf')
                error=float((pred-truth).norm()*1000)
                d.text((1010,198),'Position error',font=font(19),fill='#aac0d4');d.text((1010,230),'%.2f mm'%error,font=font(40),fill='white')
                d.text((1010,310),'GREEN / estimate',font=font(18),fill='#51e6bf');d.text((1010,342),'WHITE / simulator truth',font=font(17),fill='white')
                display.append(dict(t=t*.10002,error_mm=error,pred_mm=(pred*1000).cpu().tolist(),truth_mm=(truth*1000).cpu().tolist(),pulse=pulse,failure=bool(failed[0]),rotation=float(metrics['rotation'][0])))
                trace=display[-70:]
                for j,label in enumerate(['hand X','hand Y','hand Z']):
                    y=410+j*102;d.text((1010,y),label+' (mm)',font=font(16),fill='#aac0d4')
                    values=[v[k][j] for v in trace for k in ('truth_mm','pred_mm')];lo=min(values)-1;hi=max(values)+1
                    for key,color in [('truth_mm','white'),('pred_mm','#51e6bf')]:
                        pts=[(1012+360*i/max(1,len(trace)-1),y+80-50*(v[key][j]-lo)/(hi-lo)) for i,v in enumerate(trace)]
                        if len(pts)>1:d.line(pts,fill=color,width=2)
                d.text((1010,752),'t = %.1f s'%(t*.10002),font=font(23),fill='white')
                if pulse:d.text((1010,792),'ACTION-NOISE PULSE',font=font(19),fill='#ffbf69')
                d.rectangle((0,854,1440,900),fill='#142237')
                d.text((28,868),'Display-only estimate / no prior-to-action feedback / fixed seed, first environment, no clip selection',font=font(19),fill='#ffce88')
                writer.stdin.write(np.asarray(canvas).tobytes());frames+=1
                if t in (30,125,200):canvas.save(out/('preview-%03d.png'%t))
                if valid[0] and env.reset_buf[0]:video_active=False
                if t%50==0:print(json.dumps(dict(step=t,frames=frames,seconds=time.time()-started)),flush=True)
    finally:
        writer.stdin.close();assert writer.wait(timeout=30)==0;hook.remove()
    assert hashes==dict(actor=digest(actor),prior=digest(wm),readout=digest(head))
    save_checkpoint(out/'transitions.pt',{k:torch.stack([r[k] for r in data]) for k in data[0]})
    write_json(out/'record.json',dict(seed=a.seed,frames=frames,control_dt=.10002,steps=a.steps,envs=16,source_actor=str(actorpath),actor_sha256=file_digest(actorpath),
        head_sha256=file_digest(headpath),prior_sha256=file_digest(RELEASE),display_rows=display,reward_max_error=parity,
        weights_unchanged=True,policy_uses_estimate=False,selection='first environment, first episode, no successful-clip selection',
        intervention='command noise .3 at steps120:130, not external force',kind='Learned historical SM policy + live perception visualization; not improved control',seconds=time.time()-started))

if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
