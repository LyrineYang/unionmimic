import unittest,ast
from pathlib import Path
from types import SimpleNamespace
import torch
from unionmimic.pen_sampling import ReferenceSampler

class SamplingTests(unittest.TestCase):
    def test_outcomes_and_probabilities_match_pinned_wuji_methods(self):
        path=Path(__file__).resolve().parents[1]/'work/upstreams/dex-supermimic/src/dex_supermimic/tasks/pen_spin/mdp/commands.py'
        tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SingleHandTrackingCommand')
        names={'_configure_adaptive_sampling_bins','_bin_ids_at','_current_bin_ids','_record_bin_outcomes',
               '_update_bin_failure_scores','_smoothed_bin_failure_scores','_adaptive_sampling_probabilities'}
        methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in names]
        root=ast.Module(body=[ast.ClassDef(name='Reference',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
        ns=dict(torch=torch,dist=SimpleNamespace(is_initialized=lambda:False));exec(compile(ast.fix_missing_locations(root),str(path),'exec'),ns)
        r=ns['Reference']();r.device='cpu';r.group=SimpleNamespace(frame_dt=.02)
        r.cfg=SimpleNamespace(adaptive_sampling_bin_seconds=.5,adaptive_sampling_kernel_size=1,adaptive_sampling_kernel_decay=.8,
            trajectory_sampling_mode='adaptive',adaptive_sampling_ema_decay=.99,adaptive_sampling_uniform_mix=.2,adaptive_sampling_update_steps=40)
        lengths=torch.tensor([101,51]);r._trajectory_starts=torch.tensor([0,101]);r._trajectory_ends=lengths.cumsum(0);r._configure_adaptive_sampling_bins()
        r.bin_failure_scores=torch.zeros(6);r._pending_bin_outcomes=torch.zeros(2,6);r._steps_since_score_update=0
        r.trajectory_ids=torch.tensor([0,0,1]);r.time_steps=torch.tensor([75,50,50]);r._episode_start_steps=torch.tensor([0,25,0])
        fail=torch.tensor([True,False,True]);tm=SimpleNamespace(terminated=fail,time_outs=~fail,get_term=lambda _:torch.zeros(3,dtype=torch.bool))
        r._env=SimpleNamespace(termination_manager=tm,episode_length_buf=torch.ones(3))
        s=ReferenceSampler(lengths);s.record(r.trajectory_ids,r._episode_start_steps,r.time_steps,fail);r._record_bin_outcomes(torch.arange(3))
        torch.testing.assert_close(s.pending,r._pending_bin_outcomes)
        for _ in range(40):s.update();r._update_bin_failure_scores()
        torch.testing.assert_close(s.scores,r.bin_failure_scores)
        torch.testing.assert_close(s.probabilities(torch.tensor([0,1]))[1],r._adaptive_sampling_probabilities())
    def test_survivals_count_in_every_crossed_bin(self):
        s=ReferenceSampler(torch.tensor([101,51]));s.record(torch.tensor([0,0,1]),torch.tensor([0,25,0]),torch.tensor([75,50,50]),torch.tensor([True,False,True]))
        torch.testing.assert_close(s.pending[0],torch.tensor([1.,2.,2.,1.,1.,1.]))
        torch.testing.assert_close(s.pending[1],torch.tensor([0.,0.,0.,1.,0.,1.]))
        for _ in range(40):s.update()
        self.assertAlmostEqual(float(s.scores[3]),.01,places=6);self.assertEqual(float(s.pending.sum()),0.)
    def test_conditioned_draws_and_checkpoint(self):
        s=ReferenceSampler(torch.tensor([2,30,56]));pool=torch.tensor([1,2]);c,p=s.sample(pool,1000)
        self.assertTrue(((c==1)|(c==2)).all());self.assertTrue((p<s.lengths[c]-1).all());self.assertTrue((p>=-150).all())
        self.assertGreater(int((p<0).sum()),70)
        s.scores[2]=1;r=ReferenceSampler(s.lengths);r.load_state_dict(s.state_dict());torch.testing.assert_close(s.probabilities(pool)[1],r.probabilities(pool)[1])
        probabilities=s.probabilities(pool)[1];self.assertGreater(float(probabilities.min()),0)

if __name__=='__main__':unittest.main()
