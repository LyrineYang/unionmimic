"""Explicit review boundary for new experiments, separate from the ended run."""
import json
from pathlib import Path
from .common import ROOT, file_digest


class ReviewGate:
    def __init__(self, manifest, approval, arm):
        self.path=Path(manifest).resolve()
        self.sha=file_digest(self.path)
        self.manifest=json.loads(self.path.read_text())
        self.approval=Path(approval).resolve()
        self.arm=arm

    def __call__(self):
        if file_digest(self.path)!=self.sha:
            raise RuntimeError('Launch manifest changed after review')
        decision=json.loads(self.approval.read_text())
        if not (decision.get('approved') is True and decision.get('manifest_sha256')==self.sha
                and self.arm in decision.get('arms',[])):
            raise PermissionError('User review required before any parameter update')

    def verify_inputs(self):
        for item in self.manifest['immutable_inputs']:
            path=ROOT/item['path']
            if file_digest(path)!=item['sha256']:
                raise ValueError('Reviewed input changed: '+str(path))
        for path,sha in self.manifest['source_sha256'].items():
            if file_digest(ROOT/path)!=sha:
                raise ValueError('Reviewed source changed: '+path)

