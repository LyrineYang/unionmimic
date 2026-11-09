"""Continue this authorized calibration into the gated queue, without renewing time."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from unionmimic.common import write_json
from unionmimic.training_window import remaining_seconds

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());auth=json.loads((ROOT/cfg['authorization']).read_text())
    status_path=ROOT/cfg['status'];child=None
    def status(stage,**kw):write_json(status_path,dict(stage=stage,pid=os.getpid(),unix=time.time(),deadline_unix=auth['deadline_unix'],**kw))
    def alive(job):
        path=Path('/proc')/str(job['pid'])/'cmdline'
        return path.exists() and job['identity'].encode() in path.read_bytes()
    def stop(*_):raise InterruptedError('Stopped by user; do not restart')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(script,args):
        nonlocal child
        remaining_seconds(auth)
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)]+[str(x) for x in args],cwd=ROOT,start_new_session=True)
        while child.poll() is None:
            remaining_seconds(auth);time.sleep(5)
        if child.returncode:raise RuntimeError('%s exited %s; no blind retry'%(script,child.returncode))
    try:
        status('waiting_for_calibration')
        while any(alive(j) for j in cfg['jobs']):
            remaining_seconds(auth)
            for job in cfg['jobs']:
                if not alive(job) and not (ROOT/job['result']).exists():raise RuntimeError('Calibration exited without complete result: '+job['identity'])
            time.sleep(10)
        assert all((ROOT/j['result']).exists() for j in cfg['jobs'])
        if remaining_seconds(auth)<9*3600:raise TimeoutError('Insufficient remaining window for acceptance and 8-hour comparison')
        status('assembling_candidate')
        run('merge_pen_phase_calibration.py',['--input',ROOT/cfg['input'],'--calibration',ROOT/cfg['calibration'],
            '--lanes',3,'--prefix',ROOT/cfg['prefix'],'--out',ROOT/cfg['candidate']])
        status('gated_queue',plan=cfg['plan'])
        run('run_pen_training_queue.py',['--plan',ROOT/cfg['plan'],'--authorization',ROOT/cfg['authorization']])
        status('finished')
    except BaseException as exc:
        status('stopped' if isinstance(exc,InterruptedError) else 'blocked',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        if time.time()>=auth['deadline_unix']:
            for job in cfg['jobs']:
                if alive(job):os.kill(job['pid'],signal.SIGTERM)
if __name__=='__main__':main()
