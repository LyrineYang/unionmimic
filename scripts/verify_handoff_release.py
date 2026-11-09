"""Verify the published handoff files without GPU or training dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'work/publications/handoff-20261109/manifest.json').read_text())
    for row in manifest['files']:
        path = ROOT / row['path']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        assert path.stat().st_size == row['size'], row['path']
        assert digest.hexdigest() == row['sha256'], row['path']
    print('PASS: {} published handoff files'.format(len(manifest['files'])))


if __name__ == '__main__':
    main()
