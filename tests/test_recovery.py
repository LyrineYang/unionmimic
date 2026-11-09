import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import h5py
from unionmimic.common import file_digest
from unionmimic.recovery import validate_completed_stage
from unionmimic.telemetry import TrainingMetrics


class Recovery(unittest.TestCase):
    def test_recomputed_cache_requires_explicit_mode_and_checkpoint_provenance(self):
        import torch
        from unionmimic.common import write_json,save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'source.pt';checkpoint.write_bytes(b'original')
            report=dict(seeded_replay_exact=True,full_episode_history_used=True,
                posterior_mode='recomputed_from_full_history',recorded_posterior_reproduced=False,
                max_deter_error=None,max_logit_error=None,max_embed_error=0.,
                source_checkpoint_sha256=file_digest(checkpoint),dataset_sha256='dataset',state_model_sha256='new-state')
            write_json(root/'recovery.json',report)
            save_checkpoint(root/'cache.pt',dict(metadata=report,train={'episode_id':torch.tensor([1])},valid={'episode_id':torch.tensor([2])}))
            params=['prepare','--output',root,'--checkpoint',checkpoint]
            with self.assertRaises(AssertionError):validate_completed_stage('prepare',params)
            validate_completed_stage('prepare',params+['--recompute-state'])
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(AssertionError):validate_completed_stage('prepare',params+['--recompute-state'])

    def test_worker_skips_bad_finalizer_only_after_success(self):
        script='''
import atexit,os
from unionmimic.simulation_worker import run_isolated_simulation
atexit.register(os.abort)
def work(args):
    print("output_committed",flush=True)
run_isolated_simulation(work,None)
'''
        r=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(r.returncode,0)
        self.assertIn('output_committed',r.stdout)

    def test_worker_preserves_task_failures(self):
        script='''
from unionmimic.simulation_worker import run_isolated_simulation
def work(args):
    raise RuntimeError("actual_task_failed")
run_isolated_simulation(work,None)
'''
        r=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(r.returncode,1)
        self.assertIn('actual_task_failed',r.stderr)

    def test_recovery_rejects_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'model.pt';checkpoint.write_bytes(b'original')
            output=root/'evaluation.json'
            report=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_digest(checkpoint),
                        episodes=2,seed=123,returns=[1.,3.],mean_return=2.,lengths=[499,499],
                        falls=0,failures=0,timeouts=2,failure_rate=0.)
            output.write_text(json.dumps(report))
            params=['evaluate','--output',output,'--checkpoint',checkpoint,'--envs',2,'--seed',123]
            self.assertEqual(validate_completed_stage('evaluate',params)['episodes'],2)
            checkpoint.write_bytes(b'other')
            with self.assertRaises(AssertionError):validate_completed_stage('evaluate',params)

    def test_recovery_rejects_incomplete_hdf5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'collection.json').write_text(json.dumps(dict(episodes=1)))
            with h5py.File(root/'episodes.h5','w') as f:
                f.attrs['complete']=False
            with self.assertRaises(AssertionError):
                validate_completed_stage('collect',['collect','--output',root])

    def test_metrics_resume_preserves_update_and_event_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=TrainingMetrics(tmp,{})
            first.emit('sm',1,{'loss':1.},1000)
            first.close()
            resumed=TrainingMetrics(tmp,{},dict(started_unix=1,attempted_updates=dict(sm=1000,calibrate=2000)))
            resumed.begin_phase('sm',2)
            resumed.ingest(dict(phase='sm',step=501,actor_loss=.5),'sm',2)
            resumed.close()
            rows=[json.loads(x) for x in (Path(tmp)/'events.jsonl').read_text().splitlines()]
            self.assertEqual([x['event'] for x in rows],[1,2])
            self.assertEqual([x['step'] for x in rows],[1000,1001])


if __name__=='__main__':unittest.main()
