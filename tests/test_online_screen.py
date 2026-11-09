import copy
import unittest
import torch
from unionmimic.common import ROOT
from unionmimic.online_screen.branch import collection_steps, validate_branch
from unionmimic.online_baseline.core import Actor, Critic, Replay
from unionmimic.online_baseline.task import HandTask, PhysicalModel, DS, DC
from unionmimic.online_baseline.learner import Learner


class ScreeningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ck=torch.load(ROOT/'work/runs/sm-candidate-12h-20261102-v1/sm-detach-warm/checkpoints/segment-07.pt',map_location='cpu')

    def test_utd_exact_over_long_and_resumed_cycles(self):
        for ratio,n in [(.125,8000),(.4,2500)]:
            cfg=dict(utd=ratio,branch_initial_cycles=4800)
            seq=[collection_steps(cfg,i+4800) for i in range(1000)]
            self.assertEqual(sum(seq),n)
            for cut in [1,17,399]:
                self.assertEqual(seq[cut:],[collection_steps(cfg,i+4800) for i in range(cut,1000)])
        self.assertEqual(seq[:4],[2,3,2,3])

    def test_branch_rejects_unplanned_task_or_config_change(self):
        c=self.ck
        cfg=dict(c['config'],failure_fraction=.25,critic_norm_source='real',horizon=16,
                 utd=.4,branch_initial_cycles=c['learner']['counts']['cycles'])
        validate_branch(c,cfg,c['task_constants'])
        with self.assertRaises(ValueError):validate_branch(c,dict(cfg,batch=128),c['task_constants'])
        with self.assertRaises(ValueError):validate_branch(c,cfg,dict(c['task_constants'],latency=999))

    def test_all_eight_branch_updates_finite_and_preserve_f_during_actor(self):
        for f,n,u,h in [(0,'mixed',.125,8),(0,'mixed',.4,16),(0,'real',.125,16),(0,'real',.4,8),
                         (.25,'mixed',.125,16),(.25,'mixed',.4,8),(.25,'real',.125,8),(.25,'real',.4,16)]:
            c=self.ck;task=HandTask(c['task_constants'])
            cfg=dict(c['config'],batch=8,critic_epochs=1,failure_fraction=f,critic_norm_source=n,horizon=h)
            obj=Learner(Actor(),Critic(DS+DC),PhysicalModel(task),task,cfg)
            obj.load_state_dict(copy.deepcopy(c['learner']))
            r=Replay(c['replay']['capacity'],c['replay']['nenv'],'cpu');r.load_state_dict(copy.deepcopy(c['replay']))
            before=copy.deepcopy(obj.actor.state_dict());obj.fit(r,1)
            physics=copy.deepcopy(obj.model.state_dict());report=obj.imagine(r)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in report.values()))
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in obj.actor.state_dict().items()))
            for k,v in obj.model.state_dict().items():torch.testing.assert_close(v,physics[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
