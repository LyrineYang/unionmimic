"""Unselected initial grasp diagnostic, explicitly not a trained policy demo."""
import ctypes,os,sys
ctypes.CDLL('libEGL_nvidia.so.0',mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_NODELETE)
import isaacgym
from isaacgym import gymapi
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import argparse,json,subprocess,traceback
import torch,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
from unionmimic.pen_simulator import make_pen_simulator
from unionmimic.common import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);root=Path(a.input).resolve()
    torch.set_num_threads(4);env,cfg=make_pen_simulator(4,a.gpu,20261110,root,True)
    env.fixed_clips=True;env.clip_ids[:]=torch.tensor([0,4,8,12],device=env.device);env.task_ids[:]=1;env.reset_buf[:]=1
    cameras=[]
    for i in range(4):
        props=gymapi.CameraProperties();props.width=512;props.height=512;props.horizontal_fov=45
        camera=env.gym.create_camera_sensor(env.envs[i],props);cameras.append(camera)
        env.gym.set_camera_location(camera,env.envs[i],gymapi.Vec3(-.075,-.35,1.264),gymapi.Vec3(-.075,0,.964))
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024',
        '-r','50','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/'diagnostic.mp4')],stdin=subprocess.PIPE)
    rows=[]
    try:
        for step in range(75):
            if step:env.reset_buf.zero_()
            env.step(torch.zeros(4,22,device=env.device))
            env.gym.fetch_results(env.sim,True);env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
            canvas=Image.new('RGB',(1024,1024))
            for i,camera in enumerate(cameras):
                rgba=env.gym.get_camera_image(env.sim,env.envs[i],camera,gymapi.IMAGE_COLOR).reshape(512,512,4)
                img=Image.fromarray(rgba[:,:,:3]);draw=ImageDraw.Draw(img);draw.rectangle((0,0,512,45),fill='black')
                draw.text((5,5),'Sharpa pen / initialization only / zero command',fill='white')
                draw.text((5,25),'clip=%d  t=%.2fs  failed=%s'%(env.clip_ids[i],(step+1)*.02,bool(env.pen_failure[i])),fill='white')
                canvas.paste(img,((i%2)*512,(i//2)*512))
            encoder.stdin.write(np.asarray(canvas).tobytes())
            if step in (0,4,24,49):canvas.save(out/('frame-%03d.png'%step))
            rows.append(dict(step=step,failure=env.pen_failure.tolist(),position_m=env.pen_metrics['position_error_m'].tolist()))
    finally:encoder.stdin.close();assert encoder.wait(timeout=30)==0
    write_json(out/'record.json',dict(kind='diagnostic only, no learned policy, no selection',rows=rows,retarget=str(root)))
if __name__=='__main__':
    rc=0
    try:main()
    except BaseException:traceback.print_exc();rc=1
    sys.stdout.flush();sys.stderr.flush();os._exit(rc)
