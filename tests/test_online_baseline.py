import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from unionmimic.common import ROOT
from unionmimic.online_baseline.core import Actor, Critic, Replay
from unionmimic.online_baseline.task import HandTask, PhysicalModel, DS, DC
from unionmimic.online_baseline.learner import Learner
from unionmimic.online_baseline.curriculum import capture_curriculum, restore_curriculum, FIELDS


class BaselineAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ck=torch.load(ROOT/'work/runs/online-sm-20261102-probe02/last.pt',map_location='cpu')

    def make(self, fraction=0., norm='mixed'):
        ck=self.ck
        task=HandTask(ck['task_constants'])
        cfg=dict(ck['config'],batch=8,model_horizon=4,horizon=4,critic_epochs=1,
                 failure_fraction=fraction,critic_norm_source=norm)
        obj=Learner(Actor(),Critic(DS+DC),PhysicalModel(task),task,cfg)
        obj.load_state_dict(copy.deepcopy(ck['learner']))
        r=Replay(ck['replay']['capacity'],ck['replay']['nenv'],'cpu');r.load_state_dict(copy.deepcopy(ck['replay']))
        return obj,r

    def test_failure_mixture_targets_original_distribution_and_excludes_invalid_holdout(self):
        r=Replay(16,8,'cpu')
        for t in range(24):
            r.add(dict(episode=torch.arange(8), valid=torch.arange(8)!=7,
                failure=(torch.arange(8)==1)&(t%3==0), timeout=torch.zeros(8,dtype=torch.bool),
                value=torch.full((8,),t)))
        batch,w,stats=r.failure_sample(100000,.25)
        y=batch['failure'][0].float()
        self.assertGreater(float(y.mean()),.25)
        self.assertAlmostEqual(float((w*y).mean()),stats['failure_pool_rate'],delta=.003)
        self.assertAlmostEqual(float(w.mean()),1.,delta=.01)
        self.assertTrue((batch['episode']%5!=0).all());self.assertTrue(batch['valid'].all())
        self.assertTrue((batch['value']>=8).all())
        # An independent constant-logit BCE derivative equals sigmoid(z)-p.
        z=torch.tensor(.3,requires_grad=True)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(z.expand_as(y),y,reduction='none')*w).mean()
        derivative=torch.autograd.grad(loss,z)[0]
        self.assertAlmostEqual(float(derivative),float(z.sigmoid())-stats['failure_pool_rate'],delta=.01)
        r.data['failure'].zero_()
        _,w,stats=r.failure_sample(100,.25)
        self.assertEqual(stats['failure_mix'],0.);self.assertTrue((w==1).all())

    def test_real_only_normalizer_stays_fixed_inside_imagination_and_exact_resume(self):
        for frac in (0.,.25):
            a,r=self.make(frac,'real');a.fit(r,1)
            b,_=self.make(frac,'real');b.load_state_dict(copy.deepcopy(a.state_dict()))
            stats=copy.deepcopy(a.critic.norm.state_dict());physics=copy.deepcopy(a.model.state_dict())
            before=copy.deepcopy(a.actor.state_dict())
            rng=torch.get_rng_state();a.imagine(r);torch.set_rng_state(rng);b.imagine(r)
            for k,v in a.critic.norm.state_dict().items():torch.testing.assert_close(v,stats[k],rtol=0,atol=0)
            for k,v in a.model.state_dict().items():torch.testing.assert_close(v,physics[k],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in a.actor.state_dict().items()))
            for name in ('actor','critic','model'):
                for k,v in getattr(a,name).state_dict().items():
                    torch.testing.assert_close(v,getattr(b,name).state_dict()[k],rtol=0,atol=0)

    def test_imagination_does_not_read_future_recorded_state_reward_or_failure(self):
        a,r=self.make(.25,'real');b,r2=self.make(.25,'real')
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        r2.data['sn'].fill_(12345);r2.data['reward'].fill_(-777);r2.data['failure'].logical_not_()
        # Window=1 sampling is independent of next-step labels. Start s/h/ctx stay identical.
        rng=torch.get_rng_state();a.imagine(r);torch.set_rng_state(rng);b.imagine(r2)
        for k,v in a.actor.state_dict().items():torch.testing.assert_close(v,b.actor.state_dict()[k],rtol=0,atol=0)

    def test_policy_has_only_deployable_history_input(self):
        actor=Actor()
        self.assertEqual(actor.net[0].in_features,292)
        self.assertEqual(actor(torch.randn(4,292)).shape,(4,22))
        with self.assertRaises(RuntimeError):actor(torch.randn(4,292+DS+DC))

    def test_curriculum_restores_ring_and_level_and_rejects_bad_state(self):
        env=SimpleNamespace(device='cpu',gravity_curriculum_outcomes=torch.tensor([1.,0.,1.,1.]),
            gravity_curriculum_current_max_angle_deg=2.,gravity_curriculum_upgrade_count=20,
            gravity_curriculum_full_run_ratio=.75,gravity_curriculum_outcome_write_idx=2,
            gravity_curriculum_outcome_count=4,action_conservative_scale=1.,
            gravity_curriculum_start_max_angle_deg=1.,gravity_curriculum_max_angle_deg=4.)
        saved=capture_curriculum(env)
        for k in FIELDS:setattr(env,k,0)
        env.gravity_curriculum_outcomes.zero_();restore_curriculum(env,saved)
        self.assertEqual(env.gravity_curriculum_current_max_angle_deg,2.)
        self.assertEqual(env.gravity_curriculum_outcome_count,4)
        torch.testing.assert_close(env.gravity_curriculum_outcomes,saved['outcomes'])
        bad=copy.deepcopy(saved);bad['values']['gravity_curriculum_current_max_angle_deg']=100
        with self.assertRaises(ValueError):restore_curriculum(env,bad)

    def test_selection_prioritizes_rotation_and_failures_over_reward(self):
        spec=importlib.util.spec_from_file_location('newlane',ROOT/'scripts/online_baseline_lane.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        good=dict(episodes=[{}]*32,failures=1,mean_rotation=40,mean_return=700)
        bad=dict(good,failures=12,mean_return=1000)
        still=dict(good,mean_rotation=0,mean_return=1000)
        self.assertGreater(mod.selection_key(good),mod.selection_key(bad))
        self.assertGreater(mod.selection_key(good),mod.selection_key(still))


if __name__=='__main__':unittest.main()
