#!/usr/bin/env python3
"""Read-only, recorded-action H1/H4/H8/H16 model validation on the common parent."""
import json
from pathlib import Path
import torch
from unionmimic.common import ROOT, file_digest
from unionmimic.online_baseline.core import Actor,Critic,Replay
from unionmimic.online_baseline.task import HandTask,PhysicalModel,DS,DC,S
from unionmimic.online_baseline.learner import Learner,advance


def main():
    torch.set_num_threads(2);torch.manual_seed(99601)
    plan=json.loads((ROOT/'config/sm-screen-20261103.plan.json').read_text())
    path=Path(plan['parent_checkpoint']);ck=torch.load(path,map_location='cpu')
    task=HandTask(ck['task_constants']);model=PhysicalModel(task)
    model.load_state_dict(ck['learner']['model'])
    replay=Replay(ck['replay']['capacity'],ck['replay']['nenv'],'cpu');replay.load_state_dict(ck['replay'])
    b=replay.indices(512,16,split='valid');s,h=b['s'][0],b['history'][0]
    result=dict(parent_sha256=file_digest(path),samples=512,split='model-internal-holdout',
                input='Same recorded commands and contexts; autoregressive state prediction; no parameter updates',
                caveat='Surviving 16-step windows; excludes boundary-crossing windows, not a full-system unseen test or real-physics gradient test',horizons={})
    with torch.no_grad():
        for j in range(16):
            s,h,r,_,_,_=advance(task,model,s,h,b['ctx'][j],b['command'][j],b['scale'][j])
            if j+1 in [1,4,8,16]:
                result['horizons'][str(j+1)]=dict(normalized_state_mse=float(model.error(s,b['sn'][j])),
                    joint_rmse_rad=float((s[:,S['q']]-b['sn'][j,:,S['q']]).square().mean().sqrt()),
                    scaled_reward_rmse=float((r-b['reward'][j]).square().mean().sqrt()))
    out=ROOT/'work/analysis/sm-screen-20261103/parent-horizon-diagnostic.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
