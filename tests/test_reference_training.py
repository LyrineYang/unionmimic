"""Numeric comparisons against the pinned reference, without its MuJoCo stack."""
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from unionmimic.sm import lambda_returns,imagined_return,control_map


class ReferenceTraining(unittest.TestCase):
    def test_td_lambda_matches_pinned_reference_loop(self):
        source=Path('work/upstreams/sm-reference-audit/scripts/wm/train_yaw.py').read_text()
        begin=source.index('        Rt = V[-1]')
        end=source.index('        T = torch.stack(tgt[::-1])',begin)
        code=compile(textwrap.dedent(source[begin:end]),'pinned-reference-td-lambda','exec')
        torch.manual_seed(409)
        r=torch.randn(4,17,dtype=torch.float64);c=torch.rand_like(r);v=torch.randn(5,17,dtype=torch.float64)
        c[1,0]=0;c[:,1]=1
        context=dict(V=v,rews=r,conts=c,args=SimpleNamespace(horizon=4,gamma=.99,lam=.95))
        exec(code,context)
        expected=torch.stack(context['tgt'][::-1])
        torch.testing.assert_close(lambda_returns(r,c,v),expected,rtol=0,atol=0)

    def test_objective_matches_explicit_prefix_products(self):
        torch.manual_seed(410)
        r=torch.randn(4,11,dtype=torch.float64,requires_grad=True)
        c=torch.rand(4,11,dtype=torch.float64,requires_grad=True)
        tail=torch.randn(11,dtype=torch.float64,requires_grad=True)
        survival=torch.cat([torch.ones_like(c[:1]),c.cumprod(0)],0)
        expected=sum(.99**h*survival[h]*r[h] for h in range(4))+.99**4*survival[4]*tail
        actual=imagined_return(r,c,tail)
        torch.testing.assert_close(actual,expected)
        for a,b in zip(torch.autograd.grad(actual.sum(),(r,c,tail),retain_graph=True),
                       torch.autograd.grad(expected.sum(),(r,c,tail))):
            torch.testing.assert_close(a,b)

    def test_bptt_through_frozen_model_matches_finite_difference(self):
        # Exact smooth surrogate separates programming gradients from the
        # unverified physical accuracy of a categorical learned RSSM.
        weight=torch.tensor(.8,dtype=torch.float64,requires_grad=False)
        def objective(theta):
            s=torch.tensor([.2],dtype=torch.float64);rewards=[];cont=[]
            for _ in range(4):
                action=torch.tanh(theta*s)
                s=weight*s+.1*action
                rewards.append(-(s-.5).square());cont.append(torch.sigmoid(2-s))
            return imagined_return(rewards,cont,-s.square()).sum()
        theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
        grad=torch.autograd.grad(objective(theta),theta)[0]
        fd=(objective(theta.detach()+1e-5)-objective(theta.detach()-1e-5))/(2e-5)
        torch.testing.assert_close(grad,fd,rtol=1e-6,atol=1e-9)
        self.assertIsNone(weight.grad)

    def test_native_action_filter_and_saturation_derivative(self):
        command=torch.tensor([[.1,2.]],dtype=torch.float64,requires_grad=True)
        action,targets=control_map(command,torch.zeros_like(command),torch.zeros_like(command),.2,
                                  torch.full_like(command,-1.),torch.ones_like(command))
        torch.testing.assert_close(action,torch.tensor([[.08,.8]],dtype=torch.float64))
        grad=torch.autograd.grad(targets.sum(),command)[0]
        torch.testing.assert_close(grad,torch.tensor([[.16,0.]],dtype=torch.float64))


if __name__=='__main__':unittest.main()
