#!/usr/bin/env python3
"""Read current lane/learner evidence without polling or allocating a GPU."""
import argparse
import datetime
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
root=Path(a.run)
for path in sorted(root.glob('*/launch.json')):
    lane=path.parent;launch=json.loads(path.read_text())
    def read(name):
        target=lane/name
        return json.loads(target.read_text()) if target.exists() else {}
    state=read('lane-status.json');worker=read('learner/status.json');best=read('best.json')
    deadline=datetime.datetime.fromtimestamp(launch['deadline_unix'],datetime.timezone.utc).isoformat()
    print(json.dumps(dict(lane=lane.name,gpu=launch['gpu'],state=state.get('status'),
        phase=state.get('phase'),worker_status=worker.get('status'),worker_phase=worker.get('phase'),
        counts=worker.get('counts'),frames=worker.get('frames'),best_development=best,
        deadline_utc=deadline,error=state.get('error') or worker.get('error')),ensure_ascii=False))
