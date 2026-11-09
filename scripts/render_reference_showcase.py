"""Render pinned Wuji references with robot meshes; kinematic playback, no policy."""
import os,sys,ast,json,argparse,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import torch,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from unionmimic.motions import MotionLibrary
from unionmimic.common import file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    upstream=ROOT/'work/upstreams/dex-supermimic'
    xml=upstream/'packages/wuji-description/src/wuji_description/resources/hand2/hand2_2/body/mjcf/right.xml'
    tree=ET.parse(xml);r=tree.getroot();r.find('compiler').set('meshdir',str((xml.parent/'../meshes/right').resolve()))
    visual=ET.SubElement(r,'visual');ET.SubElement(visual,'global',offwidth='960',offheight='720')
    ET.SubElement(visual,'headlight',ambient='.4 .4 .4',diffuse='.6 .6 .6',specular='.3 .3 .3')
    ET.SubElement(visual,'rgba',haze='.06 .09 .14 1');ET.SubElement(visual,'quality',shadowsize='2048',offsamples='4')
    world=r.find('worldbody');ET.SubElement(world,'light',pos='-.2 -.4 .6',dir='.2 .4 -.8',diffuse='.8 .85 .95',castshadow='true')
    ET.SubElement(world,'geom',type='plane',pos='0 0 -.32',size='2 2 .01',rgba='.09 .12 .17 1',contype='0',conaffinity='0')
    pen=ET.SubElement(world,'body',name='reference_pen',mocap='true')
    ET.SubElement(pen,'geom',type='cylinder',size='.00745 .074',quat='.70710678 0 .70710678 0',rgba='.8 .84 .9 1',contype='0',conaffinity='0')
    for sign,color in [(-1,'.12 .8 .7 1'),(1,'1 .42 .18 1')]:
        ET.SubElement(pen,'geom',type='box',size='.018 .009 .009',pos=f'{sign*.092} 0 0',rgba=color,contype='0',conaffinity='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(r,encoding='unicode'));data=mujoco.MjData(model)
    source=ast.parse((upstream/'src/dex_supermimic/assets/wuji_hand2.py').read_text())
    mapping=next(ast.literal_eval(x.value) for x in source.body if isinstance(x,ast.AnnAssign) and getattr(x.target,'id',None)=='WUJI_HAND2_TASK_NAMES')
    addr=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in mapping['joint']]
    wrist=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'r_wrist');mujoco.mj_forward(model,data)
    wp=data.xpos[wrist].copy();wr=data.xmat[wrist].reshape(3,3).copy();wq=data.xquat[wrist].copy()
    camera=mujoco.MjvCamera();camera.lookat[:]=wp+[0,0,-.075];camera.distance=.45;camera.azimuth=95;camera.elevation=-12
    opt=mujoco.MjvOption();opt.geomgroup[2:]=0;opt.sitegroup[:]=0
    renderer=mujoco.Renderer(model,height=720,width=960)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    lib=MotionLibrary(ROOT/'work/reference-audits/pen-motion-20261103/clips');records=[]
    for group,title in zip(lib.GROUPS,['SINGLE-AXIS PEN SPIN','MULTI-AXIS PEN MANIPULATION']):
        clip=next(c for c in lib.clips if c['group']==group);video=out/(group+'.mp4')
        writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','25','-i','-','-an','-c:v','libx264','-crf','18','-threads','2','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
        try:
            for frame in range(0,clip['frames'],2):
                data.qpos[addr]=clip['source_joints'][frame].numpy();pose=clip['pose'][frame].numpy()
                data.mocap_pos[0]=wp+wr@pose[:3];mujoco.mju_mulQuat(data.mocap_quat[0],wq,pose[3:].astype(np.float64))
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera,scene_option=opt)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,960,91),fill='#101824');draw.text((26,15),title,font=font,fill='#e8f3ff')
                draw.text((26,49),'WUJI REFERENCE  |  KINEMATIC PLAYBACK  |  NOT A TRAINED SHARPA POLICY',font=small,fill='#5eead4')
                draw.rectangle((0,672,960,720),fill='#101824');draw.text((26,685),f'Original speed / 50 Hz source     {frame*.02:05.2f} s / {clip["duration"]:.2f} s',font=small,fill='#e8f3ff')
                draw.rectangle((0,716,int(960*(frame+2)/clip['frames']),720),fill='#5eead4')
                writer.stdin.write(np.asarray(image).tobytes())
                if frame in (0,200,500):image.save(out/(group+f'-{frame:04d}.png'))
        finally:writer.stdin.close();assert writer.wait(timeout=60)==0
        records.append(dict(clip=clip['id'],source_sha256=clip['sha256'],video=str(video),duration_s=clip['duration'],fps=25,playback_speed=1,policy=False,physics=False,mesh_xml_sha256=file_digest(xml)))
    renderer.close();(out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
