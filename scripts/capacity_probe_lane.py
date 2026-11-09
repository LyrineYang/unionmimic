#!/usr/bin/env python3
"""One bounded full-loop scaling trial, co-located only with our own processes."""
import argparse,os,signal,subprocess,time,json
from pathlib import Path
from unionmimic.common import ROOT,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--envs',type=int,required=True)
    ap.add_argument('--batch',type=int,required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    root=Path(a.run).resolve()
    if root.exists():raise FileExistsError(root)
    raw=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits']).decode()
    for line in raw.splitlines():
        path=Path('/proc')/line.strip()
        try:
            if path.stat().st_uid!=os.getuid():raise RuntimeError('GPU has another user process')
            cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(ROOT) not in cmd or not any(s in cmd for s in ('train_online_screen.py','evaluate_online.py')):
                raise RuntimeError('Unrelated GPU process; no co-location')
        except FileNotFoundError:pass
    root.mkdir(parents=True);started=time.time();deadline=started+8*60
    cfg=json.loads((ROOT/'config/sm-screen-20261103.plan.json').read_text())['training_config']
    cfg=dict(cfg,envs=a.envs,batch=a.batch,horizon=16,warmup_updates=40,warmup_collect_steps=0,
             deadline_unix=deadline,allocator_fraction=.35)
    write_json(root/'config.json',cfg)
    write_json(root/'probe-status.json',dict(status='running',started_unix=started,deadline_unix=deadline,gpu=a.gpu,envs=a.envs,batch=a.batch,co_resident=True))
    cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/train_online_fast.py'),'--run',str(root/'learner'),
         '--arm','sm','--gpu',str(a.gpu),'--seed','19834','--minutes','8','--cycles','24','--config',str(root/'config.json')]
    child=None;stopping=[False]
    def stop(*_):
        stopping[0]=True
        if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with open(root/'train.log','a') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=max(1,deadline-time.time()+15))
            except subprocess.TimeoutExpired:
                stop()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError('Capacity pilot exceeded deadline')
        if stopping[0]:raise InterruptedError('User stopped capacity test')
        if code:raise RuntimeError('Capacity worker failed; inspect train.log')
        st=json.loads((root/'learner/status.json').read_text());ac=json.loads((root/'learner/acceptance.json').read_text())
        assert st['status']=='segment_complete' and ac['counts']['cycles']==24
        assert ac['initial_actor_digest']!=ac['final_actor_digest']
        parity=ac['parity'];assert parity['max_action_error']<=1e-6 and parity['max_target_error']<=1e-6 and parity['max_reward_error']<=1e-4
        import torch
        from unionmimic.online_screen.branch import check_finite
        ck=torch.load(root/'learner/last.pt',map_location='cpu');check_finite(ck)
        rows=[json.loads(l) for l in (root/'learner/events.jsonl').read_text().splitlines()]
        cycles=[r for r in rows if r['kind']=='cycle'][2:]
        result=dict(status='passed',started_unix=started,ended_unix=time.time(),gpu=a.gpu,envs=a.envs,batch=a.batch,
            counts=ac['counts'],frames=ac['frames'],parity=parity,finite_checkpoint=True,
            mean_cycle_seconds=sum(r['seconds'] for r in cycles)/len(cycles),
            peak_allocated_gib=ac['cuda_peak_allocated']/1024**3,actor_start=ac['initial_actor_digest'],
            co_resident=True,control_benefit_proven=False,long_training_launched=False)
        write_json(root/'probe-status.json',result)
    except BaseException as e:
        stop();write_json(root/'probe-status.json',dict(status='stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e),started_unix=started,ended_unix=time.time()));raise

if __name__=='__main__':main()
