"""Sharpa mesh preview of retargeted references, explicitly kinematic only."""
import os,sys,json,argparse,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MUJOCO_GL','egl')
import torch,numpy as np,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from unionmimic.common import file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(a.gpu)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    root=ROOT/'work/motions/sharpa-pen-dense-merged-20261109-v1';manifest=json.loads((root/'manifest.json').read_text())
    xml=ROOT/'work/upstreams/WM-Craftnet/assets/urdf/right_sharpa_wave/right_sharpa_wave.xml'
    source=xml.read_text()
    # Upstream XML is missing one attribute separator; repair only this in-memory preview copy.
    source=source.replace('joint="right_thumb_IP"kp=', 'joint="right_thumb_IP" kp=')
    r=ET.fromstring(source);r.find('compiler').set('meshdir',str(xml.parent/'meshes'))
    visual=ET.SubElement(r,'visual');ET.SubElement(visual,'global',offwidth='960',offheight='720')
    ET.SubElement(visual,'headlight',ambient='.4 .4 .4',diffuse='.7 .7 .7',specular='.2 .2 .2')
    ET.SubElement(visual,'quality',shadowsize='2048',offsamples='4')
    world=r.find('worldbody');ET.SubElement(world,'light',pos='.2 -.3 .5',dir='-.2 .3 -.5',diffuse='.8 .85 1',castshadow='true')
    ET.SubElement(world,'geom',type='plane',pos='0 0 -.025',size='2 2 .01',rgba='.06 .09 .14 1',contype='0',conaffinity='0')
    pen=ET.SubElement(world,'body',name='preview_pen',mocap='true')
    ET.SubElement(pen,'geom',type='cylinder',size='.00745 .074',quat='.70710678 0 .70710678 0',rgba='.8 .85 .9 1',contype='0',conaffinity='0')
    for sign,color in [(-1,'.1 .85 .7 1'),(1,'1 .4 .15 1')]:
        ET.SubElement(pen,'geom',type='box',size='.018 .009 .009',pos='%g 0 0'%(sign*.092),rgba=color,contype='0',conaffinity='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(r,encoding='unicode'));data=mujoco.MjData(model)
    addr=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in manifest['joint_names']]
    assert len(addr)==22 and len(set(addr))==22
    camera=mujoco.MjvCamera();camera.lookat[:]=[.015,0,.08];camera.distance=.44;camera.azimuth=85;camera.elevation=18
    opt=mujoco.MjvOption();opt.geomgroup[2:]=0;opt.sitegroup[:]=0
    renderer=mujoco.Renderer(model,height=720,width=960)
    font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    records=[]
    for group,title in [('02_continuous_single_axis','SINGLE-AXIS REFERENCE'),('03_continuous_multi_axis','MULTI-AXIS REFERENCE')]:
        info=next(c for c in manifest['clips'] if c['id'].startswith(group))
        path=root/info['file'];assert file_digest(path)==info['retarget_sha256'];clip=torch.load(path,map_location='cpu',weights_only=False)
        writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','25','-i','-','-an','-c:v','libx264','-crf','18','-threads','2','-pix_fmt','yuv420p',str(out/(group+'.mp4'))],stdin=subprocess.PIPE)
        trail=[]
        try:
            for t in range(0,len(clip['q']),2):
                data.qpos[addr]=clip['q'][t].numpy();pose=clip['pose'][t].numpy()
                data.mocap_pos[0]=pose[:3];data.mocap_quat[0]=pose[3:];mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera,scene_option=opt)
                rot=np.zeros(9);mujoco.mju_quat2Mat(rot,pose[3:].astype(np.float64));endpoint=pose[:3]+rot.reshape(3,3)@np.array([.105,0,0]);trail.append(endpoint)
                for j,pos in enumerate(trail[-20:]):
                    scene=renderer.scene;geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0014]*3),pos,np.eye(3).ravel(),np.array([1.,.45,.2,.15+.55*j/20]));scene.ngeom+=1
                im=Image.fromarray(renderer.render());d=ImageDraw.Draw(im)
                d.rectangle((0,0,960,90),fill='#0b1420');d.text((25,12),'SHARPA 22 / '+title,font=font(23),fill='#eef5ff')
                d.text((25,52),'KINEMATIC RETARGET PREVIEW / NOT A TRAINED POLICY',font=font(19),fill='#5eead4')
                d.rectangle((0,662,960,720),fill='#0b1420');d.text((25,674),'Contact dynamics unverified  |  Original speed  |  t=%.2f s'%(t*.02),font=font(19),fill='#ffcc82')
                d.rectangle((0,716,int(960*(t+2)/len(clip['q'])),720),fill='#5eead4')
                writer.stdin.write(np.asarray(im).tobytes())
                if t in (0,200,500):im.save(out/(group+'-%04d.png'%t))
        finally:writer.stdin.close();assert writer.wait(timeout=30)==0
        records.append(dict(clip=info['id'],sha256=file_digest(path),mesh_xml_sha256=file_digest(xml),frames=len(clip['q']),
            frame_dt=.02,video_fps=25,policy=False,physics=False,contact_verified=False,selection='first clip in each pre-existing reference group',
            preview_only_asset_fix='insert missing whitespace before thumb IP actuator kp attribute; upstream file unchanged'))
    renderer.close();(out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
