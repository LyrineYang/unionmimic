import json
from pathlib import Path
import tempfile
import unittest
from unionmimic.retention import prune_bulk_outputs
from unionmimic.telemetry import TrainingMetrics


class Observability(unittest.TestCase):
    def test_retention_keeps_live_dependencies_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            old=root/'cycle-001'
            old.mkdir(parents=True)
            current=old/'current.pt'
            discard=old/'obsolete.h5'
            report=old/'validation.json'
            for p in (current,discard,report):
                p.write_text('evidence')
            outside=Path(tmp)/'outside.pt'
            outside.write_text('retain')
            (old/'link.pt').symlink_to(outside)
            recent=root/'cycle-004'
            recent.mkdir()
            (recent/'last.pt').write_text('retain')
            prune_bulk_outputs(root,4,2,[current])
            self.assertTrue(current.exists())
            self.assertTrue(report.exists())
            self.assertTrue(outside.exists())
            self.assertTrue((recent/'last.pt').exists())
            self.assertFalse(discard.exists())

    def test_training_axis_stays_monotonic_after_rejected_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger=TrainingMetrics(tmp,{'test':True})
            logger.begin_phase('sm',1)
            logger.ingest(dict(phase='sm',step=201,actor_loss=1.),'sm',1)
            logger.ingest(dict(phase='sm',step=300,actor_loss=.5),'sm',1)
            logger.begin_phase('sm',2)
            logger.ingest(dict(phase='sm',step=201,actor_loss=.4),'sm',2)
            logger.close()
            rows=[json.loads(s) for s in (Path(tmp)/'events.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows],[1,100,101])
            self.assertTrue(list((Path(tmp)/'tensorboard').glob('events.out.tfevents.*')))


if __name__=='__main__':
    unittest.main()
