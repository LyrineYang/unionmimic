"""Diagnostic multimodal trajectories from a frozen historical SM actor.

Uses the compatible 10 Hz block interface; no training, policy prior input or
50 Hz pen claim. The released WSM observes passively and keeps all weights fixed.
"""
import os,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import unionmimic.runtime
import torch,h5py
from in_hand_rotation_mjlab.world_model.policy import Actor
from unionmimic import collect as collection
from unionmimic.common import RELEASE,file_digest,write_json
from unionmimic.simulation_worker import run_isolated_simulation

def main(a):
    a.output=Path(a.output).resolve();a.actor_checkpoint=Path(a.actor_checkpoint).resolve();a.checkpoint=RELEASE
    a.envs=16;a.steps=700;a.stress_every=2;a.probe_mode='noise';a.probe_noise=a.stress_noise
    torch.set_num_threads(4)
    checkpoint=torch.load(a.actor_checkpoint,map_location='cpu');actor=Actor.from_checkpoint(checkpoint['actor']).to('cuda:%d'%a.gpu).eval();actor.requires_grad_(False)
    original=collection.make_runtime
    def runtime(*args,**kwargs):
        env,player,cfg,obs=original(*args,**kwargs)
        assert abs(env.dt*env.control_freq_inv-.10002)<1e-5
        class FrozenSMPlayer:
            def __init__(self):
                self.model=actor
                self.history=env.obs_buf[:,:73,None].transpose(1,2).expand(-1,10,-1).clone()
            @torch.no_grad()
            def get_action(self,obs,is_deterministic=True):return actor(self.history.flatten(1))
            def env_step(self,env,command):
                incoming_reset=env.reset_buf.bool().clone();result=player.env_step(env,command)
                frame=env.obs_buf[:,:73].clamp(-5,5)
                self.history=torch.cat((self.history[:,1:],frame[:,None]),1)
                self.history[incoming_reset]=frame[incoming_reset,None].expand(-1,10,-1)
                return result
        return env,FrozenSMPlayer(),cfg,obs
    collection.make_runtime=runtime
    provenance=dict(kind='diagnostic only; frozen trained SM actor plus passive frozen released WSM',actor_checkpoint=str(a.actor_checkpoint),
        actor_sha256=file_digest(a.actor_checkpoint),wsm_checkpoint=str(RELEASE),wsm_sha256=file_digest(RELEASE),
        source_sha256=file_digest(ROOT/'scripts/collect_sm_context.py'),seed=a.seed,prior_conditioned_policy=False,control_dt=.10002,
        steps=a.steps,envs=a.envs,perturbation='action noise %g after step 30 in half the environments; not an external force'%a.stress_noise,training_updates=0,
        diagnostic_state_labels_only=a.diagnostic_state)
    write_json(a.output/'provenance.json',provenance)
    collection.collect(a)
    with h5py.File(a.output/'episodes.h5','r+') as f:
        f.attrs['behavior_policy']='frozen historical SM; no WSM policy input'
        f.attrs['behavior_policy_sha256']=provenance['actor_sha256']
    write_json(a.output/'completed.json',dict(completed=True,**provenance))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--actor-checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stress-noise',type=float,default=.15)
    p.add_argument('--diagnostic-state',action='store_true')
    run_isolated_simulation(main,p.parse_args())
