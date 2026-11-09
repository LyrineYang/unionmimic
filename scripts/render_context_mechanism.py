"""A labeled offline sensor/forecast demo, not a control-benefit demonstration."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import unionmimic.runtime
import argparse,subprocess,json
import h5py,numpy as np,torch,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from unionmimic.models import load_world_model,state_model_digest
from unionmimic.common import write_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=3);p.add_argument('--out',required=True);a=p.parse_args()
    source=ROOT/'work/analysis/sm-context-trajectories-20261109-v1';out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);wm,ck,cfg=load_world_model(source/'config.yaml',gpu=a.gpu);del ck
    wm.requires_grad_(False);dev=next(wm.parameters()).device;torch.manual_seed(29)
    ix=np.arange(30,180);horizon=4
    with h5py.File(source/'episodes.h5') as f:
        assert f.attrs['state_model_sha256']==state_model_digest(wm)
        g=f['000002'];assert g.attrs['split']=='valid'
        start=dict(deter=torch.tensor(g['post_deter'][:][ix],device=dev),stoch=torch.nn.functional.one_hot(torch.tensor(g['post_index'][:][ix],device=dev).long(),32).float())
        actions=torch.tensor(g['executed_action'][:][ix[:,None]+np.arange(1,5)],device=dev)
        truth=g['prop'][:][ix+4];depth=g['depth'][:][ix];prop=g['prop'][:][ix]
    with torch.no_grad():
        fixed=wm.heads['decoder']._mlp(wm.dynamics.get_feat(start))['prop'].mode().cpu().numpy()
        state=start
        for step in range(4):state=wm.dynamics.img_step(state,actions[:,step],sample=True)
        pred=wm.heads['decoder']._mlp(wm.dynamics.get_feat(state))['prop'].mode().cpu().numpy()
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=lambda n:ImageFont.truetype(fontpath,n)
    enc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/'context-mechanism.mp4')],stdin=subprocess.PIPE)
    try:
        for k,t in enumerate(ix):
            im=Image.new('RGB',(1280,720),(16,20,29));d=ImageDraw.Draw(im)
            d.text((34,24),'UnionMimic / What survives inside imagination?',font=font(31),fill='white')
            d.text((34,76),'Frozen WSM weights. Recurrent state advances with actions.',font=font(21),fill='#a4b6ca')
            depth8=(np.clip(depth[k,:,:,0],0,1)*255).astype(np.uint8)
            di=Image.fromarray(depth8).convert('RGB').resize((480,360),Image.Resampling.NEAREST)
            im.paste(di,(34,156));d=ImageDraw.Draw(im)
            d.text((34,119),'ACTUAL WRIST DEPTH AT t',font=font(18),fill='#b9c8d9')
            d.text((34,534),'Sharpa SM behavior | 10 Hz | corner block',font=font(17),fill='#b9c8d9')
            d.text((34,563),'Held-out episode 000002, no clip search',font=font(17),fill='#b9c8d9')
            d.text((557,119),'TACTILE READOUT AT t + 0.4 s',font=font(18),fill='white')
            for j in range(5):
                y=168+j*70;d.text((552,y+10),'S%d'%(j+1),font=font(19),fill='#a4b6ca')
                for offset,value,color in [(0,float(truth[k,44+j]),'#6cc7ff'),(17,float(fixed[k,44+j]),'#8a92a2'),(34,float(pred[k,44+j]),'#62dda8')]:
                    d.rounded_rectangle((598,y+offset,1195,y+offset+11),radius=4,fill='#252d3a')
                    d.rounded_rectangle((598,y+offset,598+max(1,np.clip(value,0,1)*597),y+offset+11),radius=4,fill=color)
            d.text((557,539),'Sensor observation',font=font(17),fill='#6cc7ff')
            d.text((557,568),'Fixed start state',font=font(17),fill='#8a92a2')
            d.text((825,568),'Action-driven state',font=font(17),fill='#62dda8')
            d.text((34,628),'OFFLINE DIAGNOSTIC: future recorded actions supplied; future observations used only for scoring.',font=font(17),fill='#e9bd74')
            d.text((34,659),'Readouts are not calibrated contact probabilities. This clip does not demonstrate a better policy.',font=font(17),fill='#e9bd74')
            enc.stdin.write(np.asarray(im).tobytes())
            if k==30:im.save(out/'preview.png')
    finally:
        enc.stdin.close();assert enc.wait(timeout=30)==0
    write_json(out/'provenance.json',dict(source=str(source),source_sha256=file_digest(source/'episodes.h5'),episode='000002',start=30,end=179,
        video_fps=10,control_dt=.10002,horizon_steps=4,weights_frozen=True,policy_training=False,
        future_inputs='recorded executed actions only',future_observation_use='offline labels only',
        purpose='Mechanism visualization, not a policy-benefit result',source_script_sha256=file_digest(Path(__file__))))
if __name__=='__main__':main()
