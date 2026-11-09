"""Request an explicitly authorized in-process pilot -> long-run transition."""
import argparse,json,os,time
from pathlib import Path
from unionmimic.common import write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['promote']);ap.add_argument('--run',required=True)
    a=ap.parse_args();run=Path(a.run).resolve()
    manifest=json.loads((run/'manifest.json').read_text());status=json.loads((run/'status.json').read_text())
    assert manifest['cli']['continue_after_gate'],'No authorization for long continuation'
    assert status['status']=='running' and status['phase']=='pilot','Not a running pilot'
    assert time.time()<manifest['deadline_unix'],'Original budget expired'
    proc=Path('/proc')/str(status['pid']);assert proc.stat().st_uid==os.getuid()
    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(run) in cmd and 'train_sharpa_leap.py' in cmd,'Process identity mismatch'
    target=run/'promote-request.json'
    if target.exists():raise FileExistsError(target)
    write_json(target,dict(action='promote',explicit_user_request=True,requested_unix=time.time(),
        preserve_model_optimizer_replay_rng=True,minimum_model_warmup=300,minimum_current_policy_recollection=1,
        independent_evaluation_required=True,deadline_unix=manifest['deadline_unix']))
    print('Promotion requested; retains 300 warm-up rounds, checks and the original deadline.')


if __name__=='__main__':main()
