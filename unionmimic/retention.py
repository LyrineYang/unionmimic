"""Remove only old bulk outputs created within this training run."""
import time
from pathlib import Path
from .common import write_json


def prune_bulk_outputs(run,cycle,keep_cycles,protected_paths):
    root=Path(run).resolve()
    protected={Path(p).resolve() for p in protected_paths}
    removed=[]
    for folder in root.glob('cycle-*'):
        if folder.is_symlink() or not folder.is_dir():
            continue
        try:
            number=int(folder.name.split('-')[-1])
        except ValueError:
            continue
        if number>cycle-keep_cycles:
            continue
        for path in folder.rglob('*'):
            if path.suffix not in ('.pt','.h5') or path.is_symlink() or not path.is_file():
                continue
            resolved=path.resolve()
            if root not in resolved.parents or resolved in protected:
                continue
            size=path.stat().st_size
            path.unlink()
            removed.append(dict(path=str(path.relative_to(root)),bytes=size))
    if removed:
        write_json(root/('retention-%03d.json'%cycle),dict(time=time.time(),removed=removed,
                   freed_bytes=sum(r['bytes'] for r in removed)))
    return sum(r['bytes'] for r in removed)
