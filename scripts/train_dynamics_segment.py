"""Resumable RSSM-only fitting with an immutable representation anchor."""
import argparse,json,time
from pathlib import Path
from unionmimic.common import ROOT,write_json,save_checkpoint,digest,file_digest


def restore_dynamics(saved,fit,opt,chunks,reference,device):
    import torch
    # The trust reference remains the original representation across segments.
    body=reference['world_model']['world_model']
    fit.reference.load_state_dict({k[len('dynamics.'):]:v for k,v in body.items() if k.startswith('dynamics.')},strict=True)
    state=saved.get('dynamics_state')
    if state is None:return 0
    opt.load_state_dict(state['optimizer'])
    if 'sampler_rng' in state:
        chunks.rng.bit_generator.state=state['sampler_rng']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'],device)
    return state['step']


def train(a):
    import unionmimic.runtime
    import torch
    from unionmimic.models import load_world_model
    from unionmimic.dynamics_fit import DynamicsFit,EpisodeChunks
    from unionmimic.review_gate import ReviewGate
    from torch.utils.tensorboard import SummaryWriter
    gate=ReviewGate(a.manifest,a.approval,'dynamics');gate.verify_inputs();gate()
    c=gate.manifest['dynamics'];torch.manual_seed(a.seed)
    wm,saved,cfg=load_world_model(a.data/'config.yaml',a.checkpoint,a.gpu)
    device=next(wm.parameters()).device;fit=DynamicsFit(wm,saved)
    chunks=EpisodeChunks(a.data,digest(wm.encoder),32,a.seed)
    params=fit.parameters();opt=torch.optim.Adam(params,lr=c['lr'])
    reference=torch.load(gate.manifest['checkpoint'],map_location='cpu')
    expected_anchor=file_digest(gate.manifest['checkpoint'])
    if 'anchor_checkpoint_sha256' in saved.get('dynamics_state',{}):
        assert saved['dynamics_state']['anchor_checkpoint_sha256']==expected_anchor
    start=restore_dynamics(saved,fit,opt,chunks,reference,device)
    frozen=(digest(wm.encoder),digest(wm.heads['decoder']._cnn));initial=digest(wm.dynamics)
    writer=SummaryWriter(str(a.output/'tensorboard'));began=time.time();last=None
    with open(a.output/'metrics.jsonl','x',buffering=1) as log:
        for step in range(start+1,start+a.updates+1):
            if time.time()>=a.deadline:break
            rows=chunks.sample(c['batch'],device);opt.zero_grad(set_to_none=True)
            loss,metrics=fit.loss(rows,4,c['feature_drift_coefficient']);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,10.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite RSSM gradient')
            if time.time()>=a.deadline:break
            gate();opt.step()
            row=dict(step=step,loss=float(loss),grad_norm=float(norm),seconds=time.time()-began,**metrics)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            for k,v in row.items():
                if isinstance(v,(int,float)):writer.add_scalar('dynamics/'+k,v,step)
            writer.flush();write_json(a.output/'status.json',dict(status='running',phase='fit',updated_unix=time.time(),**row))
            if step==start+1 or step%50==0 or step==start+a.updates:
                assert frozen==(digest(wm.encoder),digest(wm.heads['decoder']._cnn))
                assert initial!=digest(wm.dynamics)
                bundle=fit.checkpoint(saved,opt,step)
                bundle['dynamics_state'].update(sampler_rng=chunks.rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),
                    anchor_checkpoint_sha256=file_digest(gate.manifest['checkpoint']))
                save_checkpoint(a.output/'last.pt',bundle);last=row
                print(json.dumps(row),flush=True)
    chunks.close();writer.close()
    if last is None:raise TimeoutError('No RSSM checkpoint before original deadline')
    write_json(a.output/'completed.json',dict(**last,restored_step=start,actor_unchanged=True,
        encoder_unchanged=True,image_decoder_unchanged=True,dynamics_changed=True,
        source_checkpoint=str(a.checkpoint),checkpoint=str(a.output/'last.pt'),
        anchor_checkpoint_sha256=file_digest(gate.manifest['checkpoint'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','data','output','manifest','approval'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--updates',type=int,default=250);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    for name in ('checkpoint','data','output','manifest','approval'):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=False);train(a)
