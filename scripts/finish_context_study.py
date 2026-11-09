"""One-shot completion/evaluation, never launches or resumes training."""
import argparse,json,time,subprocess,sys
from pathlib import Path
from unionmimic.common import write_json,file_digest
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--arm',required=True,choices=['none','fixed','rollout'])
    p.add_argument('--gpu',required=True,type=int);a=p.parse_args();run=Path(a.run).resolve()/a.arm
    manifest=json.loads((run/'manifest.json').read_text());deadline=min(1794311095.,manifest['deadline_unix']+1800)
    while time.time()<deadline:
        status=json.loads((run/'status.json').read_text())
        if status['status'] in ('failed','stopped'):raise RuntimeError(status)
        if status['status']=='completed':break
        time.sleep(10)
    else:raise TimeoutError('No completed training within original allowance')
    result=json.loads((run/'result.json').read_text());assert result['passed'] and result['actor_updates']==100
    # Wait for the original training worker to exit before evaluation on its card.
    pid=status['pid']
    for _ in range(60):
        cmd=Path('/proc/%d/cmdline'%pid)
        if not cmd.exists() or b'train_context_study.py' not in cmd.read_bytes():break
        time.sleep(1)
    else:raise RuntimeError('Training worker did not exit')
    for tag,pulse in [('clean',False),('pulse',True)]:
        assert time.time()<deadline-650,'Insufficient original evaluation allowance'
        cmd=[sys.executable,str(ROOT/'scripts/evaluate_context_study.py'),'--gpu',str(a.gpu),'--checkpoint',str(run/'final.pt'),'--out',str(run/('eval-'+tag+'.json'))]
        if pulse:cmd.append('--pulse')
        with (run/('eval-'+tag+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=650)
    write_json(run/'evaluation-completed.json',dict(completed=True,unix=time.time(),checkpoint_sha256=file_digest(run/'final.pt')))
if __name__=='__main__':main()
