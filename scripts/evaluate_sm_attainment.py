"""Evaluate immutable early checkpoints; offline evaluation costs stay separate."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from unionmimic.common import write_json,file_digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();root=Path(__file__).resolve().parents[1];run=Path(a.run).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=json.loads((run/'manifest.json').read_text())['started_unix'];candidates=sorted((run/'ckpt').glob('checkpoint-cycle-*.pt'))
    chosen=[]
    for minute in [45,60,90,120]:
        p=min(candidates,key=lambda p:abs(p.stat().st_mtime-started-minute*60))
        if p not in chosen:chosen.append(p)
    result=[]
    for p in chosen:
        label=p.stem;target=out/(label+'.json');begin=time.time()
        with (out/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(root/'scripts/evaluate_sharpa_leap.py'),'--gpu',str(a.gpu),'--checkpoint',str(p),'--out',str(target)],
                cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
        d=json.loads(target.read_text());d.pop('rows');d.update(checkpoint=p.name,checkpoint_elapsed_seconds=p.stat().st_mtime-started,offline_evaluation_wall=time.time()-begin)
        result.append(d);write_json(out/'results.json',result);print(json.dumps(d),flush=True)

if __name__=='__main__':main()
