import unittest
import numpy as np
import torch
from isaacgymenvs.utils.world_model import WorldModelAdapter
from unionmimic.sm import lambda_returns
from unionmimic.calibrate import classification_metrics
from unionmimic.prepare import future_indices


class Contracts(unittest.TestCase):
    def test_terminal_examples_survive_at_every_prediction_horizon(self):
        starts,ids,mask=future_indices(20,8,4,'cpu')
        terminal=(ids==19) & mask
        self.assertEqual(terminal.sum(0).tolist(),[1,1,1,1])
        self.assertEqual(mask[-1].tolist(),[True,False,False,False])
        self.assertTrue(bool((ids[mask] > starts[:,None].expand_as(ids)[mask]).all()))

    def test_rssm_reset_mask_does_not_mutate_actuator_commands(self):
        adapter=WorldModelAdapter.__new__(WorldModelAdapter)
        adapter.device=torch.device('cpu')
        actions=torch.ones(4,22)
        adapter.cache_prev_action(actions)
        adapter._wm_prev_action[[1,3]]=0
        self.assertTrue(torch.equal(actions,torch.ones(4,22)))

    def test_failure_stops_bootstrap_but_timeout_does_not(self):
        rewards=[torch.tensor([1.]),torch.tensor([2.])]
        values=torch.tensor([[0.],[10.],[20.]])
        failed=lambda_returns(rewards,[torch.ones(1),torch.zeros(1)],values,gamma=.5,lam=1.)
        truncated=lambda_returns(rewards,[torch.ones(1),torch.ones(1)],values,gamma=.5,lam=1.)
        torch.testing.assert_close(failed,torch.tensor([[2.],[2.]]))
        torch.testing.assert_close(truncated,torch.tensor([[7.],[12.]]))

    def test_constant_failure_prediction_has_no_discrimination(self):
        labels=np.array([0]*99+[1])
        m=classification_metrics(np.full(100,.01),labels,.01)
        self.assertEqual(m['false_positive_rate'],1.)


if __name__=='__main__':
    unittest.main()
