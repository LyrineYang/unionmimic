import copy
import unittest
import torch
from unionmimic.online_baseline.core import checked_step as reference
from unionmimic.online_fast.core import checked_step as candidate


class FastSafety(unittest.TestCase):
    def test_same_clipped_adam_updates_and_state(self):
        torch.manual_seed(129)
        a=torch.nn.Sequential(torch.nn.Linear(7,17),torch.nn.ELU(),torch.nn.Linear(17,3))
        b=copy.deepcopy(a);oa=torch.optim.Adam(a.parameters(),lr=.002);ob=torch.optim.Adam(b.parameters(),lr=.002)
        x=torch.randn(32,7)
        for i in range(4):
            na=reference(a(x).square().mean(),oa,a.parameters(),.1)
            nb=candidate(b(x).square().mean(),ob,b.parameters(),.1)
            self.assertEqual(na,nb)
            for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=0,atol=0)
    def test_rejects_nonfinite_loss_and_gradients(self):
        for step in [reference,candidate]:
            p=torch.nn.Parameter(torch.tensor(0.));o=torch.optim.Adam([p])
            with self.assertRaises(FloatingPointError):step(p*float('nan'),o,[p])
            # Finite sqrt(0) objective, infinite derivative.
            with self.assertRaises(FloatingPointError):step(p.sqrt(),o,[p])
    def test_rejects_nonfinite_parameters_after_step(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        class Corrupt:
            def zero_grad(self,**kw):p.grad=None
            def step(self):p.data.fill_(float('inf'))
        with self.assertRaises(FloatingPointError):candidate(p.square(),Corrupt(),[p])

if __name__=='__main__':unittest.main()
