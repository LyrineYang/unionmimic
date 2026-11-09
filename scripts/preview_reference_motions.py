"""Source-trajectory previews, explicitly not Sharpa policy performance."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg
import torch
from unionmimic.motions import MotionLibrary,qrot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);lib=MotionLibrary(ROOT/'work/reference-audits/pen-motion-20261103/clips')
    matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    manifest=[]
    for group in lib.GROUPS:
        index=next(i for i,c in enumerate(lib.clips) if c['group']==group);c=lib.clips[index]
        fig=plt.figure(figsize=(6,6));ax=fig.add_subplot(111,projection='3d')
        lo=c['links'].amin((0,1)).numpy()-.04;hi=c['links'].amax((0,1)).numpy()+.04
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='wrist x (m)',ylabel='wrist y (m)',zlabel='wrist z (m)')
        ax.view_init(25,-65);ax.set_box_aspect(hi-lo)
        fingers=[ax.plot([],[],[],lw=3,color=col)[0] for col in ['#DA7635','#4277AF','#328A68','#9260AC','#B55368']]
        pen=ax.plot([],[],[],lw=6,color='#222222')[0];txt=ax.text2D(.01,.97,'',transform=ax.transAxes)
        fig.suptitle(group+' / clip_001\nSOURCE REFERENCE ONLY — not a trained Sharpa policy',fontsize=10)
        writer=FFMpegWriter(fps=25,codec='libx264',extra_args=['-pix_fmt','yuv420p','-threads','2'])
        path=out/(group+'.mp4')
        with writer.saving(fig,str(path),dpi=100):
            for frame in range(0,c['frames'],2):
                links=c['links'][frame].numpy()
                for k,line in enumerate(fingers):
                    pts=links[[0]+list(range(1+4*k,5+4*k))];line.set_data_3d(pts[:,0],pts[:,1],pts[:,2])
                pose=c['pose'][frame];axis=qrot(pose[3:][None],torch.tensor([[.1,0.,0.]]))[0]
                ends=torch.stack((pose[:3]-axis,pose[:3]+axis)).numpy();pen.set_data_3d(ends[:,0],ends[:,1],ends[:,2])
                txt.set_text('t = %.2f s / %.2f s | 50 Hz source, 25 fps preview'%(frame*.02,c['duration']))
                writer.grab_frame()
                if frame==0:fig.savefig(out/(group+'.png'))
        plt.close(fig);manifest.append(dict(id=c['id'],source_sha256=c['sha256'],frames=c['frames'],duration=c['duration'],video=str(path),
            kind='source_wrist_local_reference',sharpa_retarget_verified=False,policy_rollout=False))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
