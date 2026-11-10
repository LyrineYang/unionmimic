"""Build a hash-checked task overlay from the pinned LEAP source, not a second learner."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'work/upstreams/sm-reference-audit'
OUT=ROOT/'work/runtime/leap-sharpa'


def replace(text, old, new):
    assert old in text, old
    return text.replace(old,new)


def main():
    plan=json.loads((ROOT/'config/leap-native-validation-20261103.plan.json').read_text())
    for p,h in plan['upstream']['source_sha256'].items():
        assert hashlib.sha256((UP/p).read_bytes()).hexdigest()==h,p
    pkg=OUT/'src/in_hand_rotation_mjlab/world_model';pkg.mkdir(parents=True,exist_ok=True)
    for rel in ['__init__.py','world_model/__init__.py','world_model/policy.py','world_model/replay.py']:
        src=UP/'src/in_hand_rotation_mjlab'/rel;dst=OUT/'src/in_hand_rotation_mjlab'/rel
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
    record=(UP/'src/in_hand_rotation_mjlab/world_model/record.py').read_text()
    node=next(n for n in ast.parse(record).body if isinstance(n,ast.FunctionDef) and n.name=='state_slices')
    (pkg/'layout.py').write_text('from __future__ import annotations\n\n'+ast.get_source_segment(record,node)+'\n')
    model=(UP/'src/in_hand_rotation_mjlab/world_model/model.py').read_text()
    (pkg/'model.py').write_text(replace(model,'from .record import state_slices','from .layout import state_slices'))
    source=(UP/'scripts/wm/train_yaw.py').read_text()
    fit=source[source.index('      # 1. world model'):source.index('      if n_up <= args.wm_only:')]
    fit=replace(fit,'sh, _ = wm(sh, cmap(sh[..., wm.S["cmd"]], acts[j], sh[..., wm.S["q"]]), c)',
                'sh = wm.physical_transition(sh, acts[j], c, scale=replay.plant[(g+j)%replay.L,e], external_force=snx[j,:,145:148])')
    a=fit.index('        term = (replay.done == 1).nonzero()');b=fit.index('        s1, a1, n1, c1',a)
    fit=fit[:a]+'        rr, ee = replay.failure_rows(args.wm_batch)\n'+fit[b:]
    fit=replace(fit,'cmd1 = cmap(s1[..., wm.S["cmd"]], a1, s1[..., wm.S["q"]])','cmd1 = a1')
    fit=replace(fit,'p1, _ = wm(s1, cmd1, c1)','p1 = wm.physical_transition(s1, cmd1, c1, scale=replay.plant[rr,ee], external_force=n1[:,145:148])')
    fit=replace(fit,'wm.term((wm._term_feat(p1) - wm.t_mu) / wm.t_sd).squeeze(-1)','wm.failure_logit(p1, c1)')
    fit=replace(fit,'(aux(task.aux_features(n1, cmd1, c1)).squeeze(-1) - replay.rew[rr, ee][:, 3:5].sum(-1) / aux_sd).pow(2).mean()',
                'wm.sensor_loss(s1, n1, c1)')
    fit=replace(fit,'list(wm.parameters()) + list(aux.parameters())','list(wm.parameters())')
    fit=replace(fit,'        opt_wm.step()',
                '        assert torch.isfinite(ld+lt+la) & torch.stack([torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None]).all(), "Nonfinite model update"\n'
                '        opt_wm.step()')
    actor=source[source.index('      # 2. actor through imagination'):source.index('      hist.append(st)',source.index('      # 2. actor through imagination'))]
    actor=replace(actor,'set_grad([wm, aux],','set_grad([wm],')
    actor=replace(actor,'      sh4 = replay.history("s", g, e, 4)\n      dprev = torch.stack([task.dyaw(sh4[:, i], sh4[:, i + 1]) for i in range(3)], -1)',
                  '      assert len(g)==args.starts, "Insufficient full-history actor starts"\n      dprev = torch.zeros(len(g),3,device=dev)')
    a=actor.index('        cmd = cmap(');b=actor.index('        r = sum(terms.values())',a)
    actor=actor[:a]+('        command = task.imagined_command(a, ctx)\n'
                    '        s2, logit = wm(s, command, ctx)\n'
                    '        terms, dprev = task.reward_rates(s, s2, info, dprev, ctx)\n')+actor[b:]
    a=actor.index('        lag = torch.rand(');b=actor.index('        s = s2',a)
    actor=actor[:a]+('        frame, s2 = task.imagined_frame(s2, ctx)\n'
                    '        frames = torch.cat([frames[:, 1:], frame[:, None]], 1)\n')+actor[b:]
    actor=replace(actor,'      if ok:\n        opt_pi.step()',
                  '      assert ok and torch.isfinite(loss_pi) and torch.isfinite(gn), "Nonfinite actor update"\n      opt_pi.step()')
    actor=replace(actor,'          opt_v.step()',
                  '          assert torch.isfinite(l) & torch.stack([torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None]).all(), "Nonfinite critic update"\n          opt_v.step()')
    def wrap(name,args,body):
        # Original inner block has six spaces; function body has two.
        return '\ndef '+name+'('+args+'):\n  st = {}\n'+''.join(line[4:]+'\n' for line in body.splitlines())+'  return st\n'
    generated=('import math\nimport torch\nimport torch.nn.functional as F\nH_OBS=10\n'
               'def set_grad(mods,flag):\n  for m in mods:\n    for p in m.parameters():p.requires_grad_(flag)\n')
    generated+=wrap('fit_update','wm,err,replay,opt_wm,args,dev,N,norm_c',fit)
    generated+=wrap('actor_critic_update','actor,critic,wm,task,replay,opt_pi,opt_v,args,n_up,dev,norm_c,dt,r_max',actor)
    compile(generated,'native_update.py','exec')
    (OUT/'native_update.py').write_text(generated)
    (OUT/'training-overlay.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),generated.splitlines(True),fromfile='pinned/train_yaw.py',tofile='generated/native_update.py')))
    manifest=dict(upstream_commit=plan['upstream']['commit'],build_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=plan['upstream']['source_sha256'],generated_sha256={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*.py')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared pinned LEAP pure modules and task-adapted update blocks:',OUT)


if __name__=='__main__':main()
