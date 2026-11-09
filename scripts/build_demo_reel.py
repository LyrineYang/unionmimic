"""A labeled presentation reel; no generated motion or relabeled policy results."""
import json,subprocess
from pathlib import Path
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/demos/unionmimic-showcase-20261109'
OUT.mkdir(parents=True,exist_ok=False)
font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
scenes=[
 ('sharpa-leap-z-20261104-v2/demo.mp4','01 / LEARNED CONTROL','Sharpa + SuperMimic','Trained policy, fixed-seed rollout',
  ['22-joint Sharpa hand','Object z-axis rotation','Random actor initialization','Pinned LEAP learner'],8.,'Trained policy / historical diagnostic task'),
 ('wuji-mesh-showcase-20261109-v1/02_continuous_single_axis.mp4','02 / REFERENCE MOTION','Single-axis pen spin','Wuji robot mesh + recorded joint motion',
  ['Original 50 Hz reference','Playback at original speed','13 clips in the source library','Sharpa retarget under validation'],8.,'Kinematic reference playback / no learned policy'),
 ('wuji-mesh-showcase-20261109-v1/03_continuous_multi_axis.mp4','03 / REFERENCE MOTION','Multi-axis manipulation','Wuji robot mesh + recorded joint motion',
  ['Continuous orientation changes','Same source and joint ordering','Target for Sharpa SM / PPO','Next: recurrent WSM context'],8.,'Kinematic reference playback / no learned policy')]
encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r','25','-i','-','-an','-c:v','libx264','-crf','19','-threads','4','-pix_fmt','yuv420p',str(OUT/'showcase.mp4')],stdin=subprocess.PIPE)
records=[]
try:
 for si,(file,kicker,title,subtitle,bullets,duration,label) in enumerate(scenes):
  path=ROOT/'work/demos'/file;reader=imageio_ffmpeg.read_frames(str(path));meta=next(reader);index=-1;frame=None
  try:
   for f in range(int(duration*25)):
    wanted=int(f/25*meta['fps'])
    while index<wanted:
     frame=Image.fromarray(np.frombuffer(next(reader),np.uint8).reshape(meta['size'][1],meta['size'][0],3));index+=1
    canvas=Image.new('RGB',(1600,900),'#0c1421');d=ImageDraw.Draw(canvas)
    d.text((48,28),'UNIONMIMIC',font=font(25),fill='#59e6cf');d.text((48,78),title,font=font(42),fill='#edf5ff')
    d.text((48,139),subtitle,font=font(21),fill='#a8b9ce')
    if si==0:picture=frame.crop((0,50,640,480)).resize((960,645))
    else:picture=frame.crop((0,92,960,672)).resize((960,580))
    canvas.paste(picture,(40,200));d.text((1050,218),kicker,font=font(20),fill='#59e6cf')
    for j,bullet in enumerate(bullets):d.text((1050,290+61*j),bullet,font=font(20),fill='#edf5ff')
    d.text((1050,620),'RESEARCH MAINLINE',font=font(19),fill='#59e6cf')
    for j,line in enumerate(['Sharpa adaptation','Strong SM baseline','Matched PPO control','SM + WSM context']):d.text((1050,662+j*35),line,font=font(20),fill='#a8b9ce')
    d.rectangle((0,852,1600,900),fill='#152337');d.text((40,865),label+'  |  Original speed',font=font(18),fill='#edf5ff')
    d.rectangle((0,896,int(1600*(si+f/(25*duration))/3),900),fill='#59e6cf')
    encoder.stdin.write(np.asarray(canvas).tobytes())
    if f==75:canvas.save(OUT/f'preview-{si}.png')
  finally:reader.close()
  records.append(dict(source=str(path),start_s=0,duration_s=duration,playback_speed=1,label=label))
finally:encoder.stdin.close();assert encoder.wait(timeout=60)==0
(OUT/'manifest.json').write_text(json.dumps(dict(video=str(OUT/'showcase.mp4'),clips=records),indent=2)+'\n')
