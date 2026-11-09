"""Record a fixed, unselected Sharpa SM episode with a separate simulator."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from isaacgym import gymapi
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import argparse,json,subprocess,time,traceback
from unittest.mock import patch
import numpy as np
import imageio_ffmpeg
import torch
from PIL import Image,ImageDraw
from in_hand_rotation_mjlab.world_model.policy import Actor
from unionmimic.online_baseline import simulator as sim
from unionmimic.common import file_digest,write_json

class RenderRotation(sim.StudyRotation):
    def __init__(self,cfg,*args,**kw):
        cfg['enableCameraSensors']=True
        super().__init__(cfg,*args,**kw)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--seed',type=int,default=20261106)
    ap.add_argument('--steps',type=int,default=300);a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    ckpath=Path(a.checkpoint).resolve();ck=torch.load(ckpath,map_location='cpu');torch.set_num_threads(4)
    actor=Actor.from_checkpoint(ck['actor']).to('cuda:%d'%a.gpu).eval()
    with patch.object(sim,'StudyRotation',RenderRotation):env,task,_=sim.make_simulator(4,a.gpu,a.seed)
    assert env.graphics_device_id>=0
    props=gymapi.CameraProperties();props.width=640;props.height=640;props.horizontal_fov=45
    camera=env.gym.create_camera_sensor(env.envs[0],props)
    env.gym.set_camera_location(camera,env.envs[0],gymapi.Vec3(-.075,-.35,1.264),gymapi.Vec3(-.075,0,.964))
    dt=task.cfg['control_dt'];fps=1/dt
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x640',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/'demo.mp4')],stdin=subprocess.PIPE)
    h=env.obs_buf[:,:73,None].transpose(1,2).expand(-1,10,-1).clone();rows=[]
    try:
        with torch.no_grad():
            for step in range(a.steps):
                invalid=env.reset_buf.bool().clone();s=sim.state(env);c=sim.context(env)
                env.step(actor(h.flatten(1)));sn=sim.state(env);r,f,to,_,m=task.reward(s,sn,c)
                frame=env.obs_buf[:,:73].clamp(-5,5);h=torch.cat((h[:,1:],frame[:,None]),1)
                h[invalid]=frame[invalid,None].expand(-1,10,-1)
                env.gym.fetch_results(env.sim,True);env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
                rgba=env.gym.get_camera_image(env.sim,env.envs[0],camera,gymapi.IMAGE_COLOR).reshape(640,640,4)
                img=Image.fromarray(rgba[:,:,:3]);draw=ImageDraw.Draw(img)
                draw.rectangle((0,0,640,48),fill=(15,15,20));draw.text((10,8),'Sharpa22 / SM policy / z rotation / fixed seed %d'%a.seed,fill='white')
                draw.text((10,27),'t=%.2fs  episode ends on failure; no clip selection'%(step*dt),fill='white')
                encoder.stdin.write(np.asarray(img).tobytes())
                if step==20:img.save(out/'preview.png')
                rows.append(dict(step=step,reward=float(r[0]),rotation=float(m['rotation'][0]),failure=bool(f[0]),timeout=bool(to[0])))
                if not invalid[0] and env.reset_buf[0]:break
    finally:
        encoder.stdin.close();assert encoder.wait(timeout=30)==0
    write_json(out/'record.json',dict(checkpoint=str(ckpath),sha256=file_digest(ckpath),seed=a.seed,
        frames=len(rows),control_dt=dt,selection='first env, first episode, no selection',rows=rows))
    print(str(out/'demo.mp4'),flush=True)

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
