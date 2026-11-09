import json
import tempfile
import unittest
from pathlib import Path
from unionmimic.common import file_digest
from unionmimic.review_gate import ReviewGate
from unionmimic.prepare import verify_state_provenance


class ProvenanceAndReview(unittest.TestCase):
    def test_stale_rssm_requires_explicit_recompute(self):
        with self.assertRaises(ValueError):verify_state_provenance('encoder:old','encoder:new')
        verify_state_provenance('encoder:old','encoder:new',True)

    def test_changed_encoder_rejects_even_recompute(self):
        with self.assertRaises(ValueError):verify_state_provenance('old:rssm','new:rssm',True)

    def test_review_denies_unapproved_wrong_arm_and_modified_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            m=Path(folder)/'manifest.json';a=Path(folder)/'approval.json';m.write_text('{}')
            decision=dict(approved=False,manifest_sha256=file_digest(m),arms=['ppo'])
            a.write_text(json.dumps(decision));gate=ReviewGate(m,a,'ppo')
            with self.assertRaises(PermissionError):gate()
            decision['approved']=True;a.write_text(json.dumps(decision));gate()
            with self.assertRaises(PermissionError):ReviewGate(m,a,'dynamics')()
            m.write_text('{"changed":true}')
            with self.assertRaises(RuntimeError):gate()


if __name__=='__main__':unittest.main()
