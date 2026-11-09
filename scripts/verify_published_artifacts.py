"""Verify the committed result/demo release without a simulator or GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'work/publications/results-demo-20261109/manifest.json'


def main():
    manifest = json.loads(MANIFEST.read_text())
    for item in manifest['files']:
        path = (ROOT/item['path']).resolve()
        assert ROOT in path.parents, item['path']
        data = path.read_bytes()
        assert len(data) == item['bytes'], item['path']
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
    print('Verified %d published artifacts, %.2f MiB.' % (
        len(manifest['files']), sum(x['bytes'] for x in manifest['files'])/2**20))


if __name__ == '__main__':
    main()
