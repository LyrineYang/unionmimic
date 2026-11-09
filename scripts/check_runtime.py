#!/usr/bin/env python
"""Inspect/lock the isolated existing runtime without installing packages.

No environment variables or credential contents are included in the report.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import site
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def inventory():
    distributions={d.metadata['Name'].lower():d.version for d in metadata.distributions() if d.metadata['Name']}
    return dict(python=platform.python_version(),executable=sys.executable,prefix=sys.prefix,
        base_prefix=sys.base_prefix,user_site_enabled=site.ENABLE_USER_SITE,
        packages=dict(sorted(distributions.items())),
        local_dependencies={'isaacgym':str(ROOT/'work/isaacgym/python'),
                            'wm_craftnet':str(ROOT/'work/upstreams/WM-Craftnet')})

def main():
    p=argparse.ArgumentParser();p.add_argument('--write-lock',action='store_true');args=p.parse_args()
    current=inventory()
    assert Path(sys.prefix).resolve()==Path(os.environ.get('UNIONMIMIC_VENV_DIR',ROOT/'.venv')).resolve()
    assert sys.prefix!=sys.base_prefix and not site.ENABLE_USER_SITE
    assert current['python']=='3.8.20'
    assert current['packages']['torch']=='2.1.0+cu118'
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    path=ROOT/'config/runtime-lock.json'
    if args.write_lock:
        if path.exists():raise FileExistsError('Inspect existing lock before replacing it')
        path.write_text(json.dumps(current,indent=2)+'\n')
    else:
        lock=json.loads(path.read_text())
        assert lock['python']==current['python'] and lock['packages']==current['packages'],'Runtime drift'
    print(json.dumps(dict(status='passed',packages=len(current['packages']),python=current['python'],
                         isolated=True,lock_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))

if __name__=='__main__':main()
