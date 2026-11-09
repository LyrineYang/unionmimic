import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from unionmimic.common import write_json,file_digest
from unionmimic.comparison import budget,deadlines,verify_receipt,aggregate,stable,expected_worker_deadline
from scripts.train_ppo_segment import restore_ppo
from unionmimic.ppo import policy_command
from rl_games.common.a2c_common import ContinuousA2CBase
from scripts.train_dynamics_segment import restore_dynamics
import numpy as np


class Comparison(unittest.TestCase):
    def test_deadline_does_not_hide_worker_crash_or_early_failure(self):
        message='TimeoutError: No PPO update completed within the remaining budget'
        self.assertTrue(expected_worker_deadline(1,message,100,100))
        self.assertFalse(expected_worker_deadline(1,message,99,100))
        self.assertFalse(expected_worker_deadline(-11,message,101,100))
        self.assertFalse(expected_worker_deadline(1,'FloatingPointError: NaN gradient',101,100))

    def test_dynamics_resume_preserves_optimizer_sampler_and_original_anchor(self):
        torch.manual_seed(13)
        current=torch.nn.Linear(3,2);anchor=copy.deepcopy(current)
        opt=torch.optim.Adam(current.parameters(),lr=1e-5);x=torch.randn(4,3)
        def step(m,o):
            o.zero_grad();m(x).square().sum().backward();o.step()
        step(current,opt)
        rng=np.random.default_rng(91);rng.random(3)
        saved=dict(dynamics_state=copy.deepcopy(dict(step=1,optimizer=opt.state_dict(),
            sampler_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.get_rng_state())))
        clone=copy.deepcopy(current);other=torch.optim.Adam(clone.parameters(),lr=.1)
        fit=SimpleNamespace(reference=copy.deepcopy(current));chunks=SimpleNamespace(rng=np.random.default_rng(0))
        reference={'world_model':{'world_model':{'dynamics.'+k:v for k,v in anchor.state_dict().items()}}}
        with patch('torch.cuda.set_rng_state'):
            self.assertEqual(restore_dynamics(saved,fit,other,chunks,reference,'cpu'),1)
        np.testing.assert_array_equal(rng.random(10),chunks.rng.random(10))
        for a,b in zip(anchor.parameters(),fit.reference.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        step(current,opt);step(clone,other)
        for a,b in zip(current.parameters(),clone.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_upstream_action_clipping_precedes_environment_noise(self):
        agent=SimpleNamespace(clip_actions=True,is_tensor_obses=True,
            actions_low=torch.tensor([-1.,-1.]),actions_high=torch.tensor([1.,1.]))
        agent.preprocess_actions=lambda actions:ContinuousA2CBase.preprocess_actions(agent,actions)
        raw=torch.tensor([[1.2,-1.2]])
        command=policy_command(agent,raw)
        torch.testing.assert_close(command,torch.tensor([[1.,-1.]]))
        # Inward noise must move a saturated command back inside the bounds.
        noise=torch.tensor([[-.1,.1]])
        torch.testing.assert_close((command+noise).clamp(-1,1),torch.tensor([[.9,-.9]]))
        self.assertFalse(torch.equal((raw+noise).clamp(-1,1),(command+noise).clamp(-1,1)))
        torch.testing.assert_close(raw,torch.tensor([[1.2,-1.2]]))

    def plan(self):
        return dict(started_unix=1000,prior_gpu_hours=4,lanes=[dict(gpu=i,train_hours=8.5,total_hours=11,prior_card_hours=1) for i in range(4)])

    def test_budget_includes_preparation_and_does_not_reset_on_resume(self):
        p=self.plan();budget(p)
        with patch('time.time',return_value=99999):self.assertEqual(deadlines(p,p['lanes'][0]),(31600,40600))
        for changes in ({'train_hours':7.9},{'total_hours':11.1},{'prior_card_hours':1.1}):
            q=copy.deepcopy(p);q['lanes'][0].update(changes)
            with self.assertRaises(AssertionError):budget(q)

    def test_receipt_rejects_corruption_and_changed_command(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'checkpoint';f.write_text('original');r=Path(d)/'receipt.json'
            write_json(r,dict(command=['train'],artifacts={str(f):file_digest(f)}))
            verify_receipt(r,['train'])
            with self.assertRaises(ValueError):verify_receipt(r,['different'])
            f.write_text('changed')
            with self.assertRaises(ValueError):verify_receipt(r,['train'])

    def test_gate_does_not_accept_rotation_regression_or_failures(self):
        base=dict(mean_return=100,rotation_radians=20,failures=0)
        self.assertTrue(stable(base,base))
        for change in ({'rotation_radians':18},{'mean_return':94.9},{'failures':1}):
            self.assertFalse(stable(dict(base,**change),base))

    def test_ppo_optimizer_resume_matches_next_update(self):
        torch.manual_seed(9)
        actor=torch.nn.Linear(3,2);critic=torch.nn.Linear(3,1)
        ao=torch.optim.Adam(actor.parameters(),lr=2e-5);co=torch.optim.Adam(critic.parameters(),lr=3e-4)
        x=torch.randn(5,3)
        def step(m,o):
            o.zero_grad();m(x).square().sum().backward();o.step()
        step(actor,ao);step(critic,co)
        state=copy.deepcopy(dict(critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),
            actor_lr=2e-5,actor_steps=7,critic_steps=7,epoch=3,physical_frames=20480,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.get_rng_state()))
        a2=copy.deepcopy(actor);c2=copy.deepcopy(critic)
        ao2=torch.optim.Adam(a2.parameters(),lr=.1);co2=torch.optim.Adam(c2.parameters(),lr=.1)
        agent=SimpleNamespace(optimizer=ao2,ppo_device='cpu',optimizer_steps=0)
        agent.update_lr=lambda lr:[g.update(lr=lr) for g in ao2.param_groups]
        with patch('torch.cuda.set_rng_state'):
            result=restore_ppo({'ppo_state':state},agent,c2,co2)
        self.assertEqual(result,(3,20480,7));self.assertEqual(agent.optimizer_steps,7)
        step(actor,ao);step(critic,co);step(a2,ao2);step(c2,co2)
        for left,right in ((actor,a2),(critic,c2)):
            for a,b in zip(left.parameters(),right.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
