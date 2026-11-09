import unittest
import torch
from unionmimic.ppo import generalized_advantage,gaussian_kl


class PPOBoundaries(unittest.TestCase):
    def test_unchanged_small_sigma_policy_has_zero_kl(self):
        mu=torch.zeros(8,22);sigma=torch.full_like(mu,.02)
        self.assertEqual(float(gaussian_kl(mu,sigma,mu,sigma)),0.)

    def test_gaussian_kl_known_mean_shift(self):
        mu=torch.zeros(8,1);sigma=torch.full_like(mu,.1)
        self.assertAlmostEqual(float(gaussian_kl(mu,sigma,mu+.1,sigma)),.5,places=6)

    def test_timeout_uses_terminal_next_value_and_cuts_episode_recursion(self):
        r=torch.tensor([[1.],[1000.],[2.]])
        v=torch.zeros_like(r);n=torch.tensor([[10.],[9000.],[20.]])
        failure=torch.tensor([[False],[False],[True]])
        timeout=torch.tensor([[True],[False],[False]])
        valid=torch.tensor([[True],[False],[True]])
        adv,_=generalized_advantage(r,v,n,failure,timeout,valid,gamma=.5,lam=1.)
        torch.testing.assert_close(adv,torch.tensor([[6.],[0.],[2.]]))

    def test_failure_wins_if_timeout_also_true(self):
        r=torch.ones(1,1);v=torch.zeros(1,1);n=torch.full((1,1),100.)
        true=torch.ones(1,1,dtype=torch.bool)
        adv,_=generalized_advantage(r,v,n,true,true,true)
        torch.testing.assert_close(adv,r)

    def test_open_sequence_bootstraps_and_recurses(self):
        r=torch.tensor([[1.],[2.]]);v=torch.tensor([[3.],[4.]])
        n=torch.tensor([[4.],[10.]]);false=torch.zeros(2,1,dtype=torch.bool)
        adv,ret=generalized_advantage(r,v,n,false,false,~false,gamma=.5,lam=1.)
        torch.testing.assert_close(ret,torch.tensor([[4.5],[7.]]))
        torch.testing.assert_close(adv,ret-v)


if __name__=='__main__':unittest.main()
