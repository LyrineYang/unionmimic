#!/usr/bin/env python3
"""A user systemd timer dispatches bounded checks to the existing Codex chat."""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
from unionmimic.common import ROOT,write_json

UNIT='unionmimic-training-watch-four-gpu-20261101'
THREAD='01a191f4-d5b1-76d1-94e7-3052b642d139'
CODEX='/home/junqiy/.local/bin/codex'


def stop_timer(config):
    # Never stop another service, or the currently executing dispatch service.
    assert config['unit']==UNIT
    result=subprocess.run(['systemctl','--user','stop',UNIT+'.timer'],capture_output=True,text=True)
    if result.returncode:
        state=subprocess.run(['systemctl','--user','show',UNIT+'.timer','--property=LoadState','--value'],
                             capture_output=True,text=True)
        if state.stdout.strip()!='not-found':
            raise RuntimeError(result.stderr.strip())


def schedule_slots(now,deadline):
    # Avoid a near-immediate scheduled tick; installation has a manual probe.
    t=dt.datetime.fromtimestamp(now,dt.timezone.utc).replace(minute=45,second=0,microsecond=0)
    while t.timestamp()<now+1800 or t.hour%2:
        t+=dt.timedelta(hours=1)
    slots=[]
    while t.timestamp()<=deadline+2*3600:
        slots.append(t.timestamp())
        if t.timestamp()>=deadline:
            break
        t+=dt.timedelta(hours=2)
    return slots


def snapshot(run):
    rows=[]
    for gpu in range(4):
        p=run/('gpu%d'%gpu)/'status.json'
        try:
            s=json.loads(p.read_text())
            rows.append(dict(gpu=gpu,**{k:s.get(k) for k in
                ('status','pid','cycle','phase','updated_unix','attempted_updates','error')}))
        except (OSError,ValueError) as exc:
            rows.append(dict(gpu=gpu,status='unreadable',error=type(exc).__name__))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['install','tick','check-now','status','disable'])
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve();folder=run/'watchdog'
    folder.mkdir(parents=True,exist_ok=True)
    config_path=folder/'schedule.json'
    with open(folder/'dispatch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='install':
            if config_path.exists():
                previous=json.loads(config_path.read_text())
                assert not previous['active'], 'An active schedule already exists'
                write_json(folder/('schedule-before-install-%d.json'%time.time()),previous)
            states=[json.loads((run/('gpu%d'%g)/'status.json').read_text()) for g in range(4)]
            deadline=max(s['started_unix']+12*3600 for s in states)
            now=time.time(); assert now<deadline
            slots=schedule_slots(now,deadline)
            command=['systemd-run','--user','--unit='+UNIT,
                     '--description=UnionMimic two-hour training check',
                     '--on-calendar=*-*-* 00/2:45:00 UTC',
                     '--timer-property=AccuracySec=1s',
                     '--property=WorkingDirectory='+str(ROOT),
                     '--property=TimeoutStartSec=90s','--service-type=oneshot',
                     '/bin/bash',str(ROOT/'scripts/run_training_check.sh'),'tick','--run',str(run)]
            config=dict(active=True,thread_id=THREAD,run=str(run),created_unix=now,deadline_unix=deadline,
                        slots=slots,interval_seconds=7200,timezone='UTC',display_timezone='Asia/Shanghai',
                        scheduler='systemd-user',unit=UNIT,install_command=shlex.join(command),
                        delivered_slots=[],delivery='codex queue --resume',
                        prompt=str(ROOT/'config/training-watch-prompt.txt'))
            write_json(config_path,config)
            try:
                subprocess.run(command,check=True)
            except BaseException:
                config['active']=False
                write_json(config_path,config)
                raise
        else:
            config=json.loads(config_path.read_text())
        if args.action=='disable':
            stop_timer(config)
            config.update(active=False,disabled_unix=time.time())
            write_json(config_path,config)
        elif args.action in ('tick','check-now') and config['active']:
            now=time.time()
            due=[s for s in config['slots'] if s<=now and s not in config['delivered_slots']]
            if args.action=='tick' and not due:
                return
            slot=max(due) if due and args.action=='tick' else now
            event=('scheduled-' if args.action=='tick' else 'manual-')+str(int(slot))
            rows=snapshot(run)
            message=Path(config['prompt']).read_text()+'\n\n巡检事件：'+event+'\n触发时状态快照：\n'+json.dumps(rows,ensure_ascii=False)
            # Public CLI preserves this chat's model, context and permission
            # settings; it queues instead of starting a competing agent.
            result=subprocess.run([CODEX,'queue','--thread',config['thread_id'],'--message',message,'--resume'],
                                  cwd=ROOT,capture_output=True,text=True,timeout=60)
            record=dict(event=event,time=now,returncode=result.returncode,stdout=result.stdout,
                        stderr=result.stderr,snapshot=rows)
            write_json(folder/(event+'-delivery.json'),record)
            if result.returncode:
                raise RuntimeError('Codex queue failed; see '+str(folder/(event+'-delivery.json')))
            if args.action=='tick':
                config['delivered_slots'].extend(s for s in due if s not in config['delivered_slots'])
            config['last_delivery']=record
            # Final check is already queued. Stop only this training's timer.
            if slot>=config['deadline_unix']:
                stop_timer(config)
                config.update(active=False,final_check_enqueued=True)
            write_json(config_path,config)
        zone=dt.timezone(dt.timedelta(hours=8))
        print(json.dumps(dict(active=config['active'],thread_id=config['thread_id'],
              checks=[dt.datetime.fromtimestamp(s,zone).isoformat() for s in config['slots']],
              configuration=str(config_path),last_delivery_code=config.get('last_delivery',{}).get('returncode')),indent=2))


if __name__=='__main__':main()
