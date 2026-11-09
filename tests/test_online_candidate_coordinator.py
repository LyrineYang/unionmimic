import unittest
from pathlib import Path

class CandidateCoordinator(unittest.TestCase):
    def test_long_lane_dispatch_and_final_test_share_original_deadline(self):
        import importlib.util
        import hashlib
        import json
        import tempfile
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('lane_test','scripts/online_candidate_lane.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'study/sm-seed2';clock=[1000.];calls=[];training_calls=[0]
            plan=dict(source_sha256={},acceptance={'engineering_passed':True},
                arms={'sm-seed2':dict(algorithm='sm',seed=92002,gpu=1,hours=12,cycles_per_segment=600,config_overrides=dict(warmup_updates=3000,warmup_collect_steps=8))},
                training_config={},development=dict(reserve_seconds=360,envs=16,batches=2,seed=94001),
                locked_test=dict(reserve_seconds=1800,envs=20,batches=10,seed=95001,episodes_each=200))
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            approval=root/'approval.json';approval.write_text(json.dumps(dict(approved=True,
                plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())))
            class Child:
                pid=123
                def __init__(self,cmd,**kwargs):
                    calls.append(cmd)
                    if cmd[1].endswith('train_online_candidate.py'):
                        training_calls[0]+=1
                        self_case.assertEqual(cmd[cmd.index('--arm')+1],'sm')
                        self_case.assertEqual(json.loads((run/'config.json').read_text())['warmup_updates'],3000)
                        self_case.assertEqual(json.loads((run/'config.json').read_text())['warmup_collect_steps'],8)
                        self_case.assertEqual(cmd[cmd.index('--seed')+1],'92002')
                        learner=run/'learner';learner.mkdir(exist_ok=True)
                        (learner/'last.pt').write_text('test-checkpoint')
                        state='segment_complete' if training_calls[0]==1 else 'completed'
                        (learner/'status.json').write_text(json.dumps({'status':state}))
                        clock[0]=4000. if training_calls[0]==1 else 42400.1
                    else:
                        dest=Path(cmd[cmd.index('--output')+1]);dest.mkdir(parents=True)
                        n=200 if dest.name=='locked-test' else 32
                        (dest/'evaluation.json').write_text(json.dumps(dict(episodes=[{}]*n,mean_return=3.,mean_rotation=1.,failures=0)))
                        clock[0]+=100.
                def wait(self,timeout=None):return 0
                def poll(self):return 0
            self_case=self
            with patch.object(module,'ROOT',root),patch.object(module.time,'time',side_effect=lambda:clock[0]), \
                 patch.object(module.subprocess,'Popen',Child),patch.object(module.signal,'signal'), \
                 patch('sys.argv',['lane','--plan',str(plan_path),'--approval',str(approval),'--arm','sm-seed2','--run',str(run)]):
                module.main()
            state=json.loads((run/'lane-status.json').read_text())
            self.assertEqual(state['status'],'completed');self.assertEqual(state['deadline_unix'],44200.)
            self.assertFalse(state['locked_test_pending']);self.assertEqual(len(calls),4)
            self.assertIn('--resume',calls[2]);self.assertLess(clock[0],state['deadline_unix'])
