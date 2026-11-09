import sys,unittest,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
from unionmimic.leap_sharpa.paired_ppo import features,gae,actor_loss,critic_loss
from unionmimic.leap_sharpa.model import SharpaModel,SharpaTask

class PairedPPOTests(unittest.TestCase):
    def test_same_privileged_features_as_sm(self):
        c=json.loads((ROOT/'work/runs/sharpa-leap-operator-20261103-v3/task-constants.json').read_text())
        m=SharpaModel(c);task=SharpaTask(m,c,'cpu');s=torch.randn(8,177);ctx=torch.randn(8,100)
        s[:,69:73]=torch.nn.functional.normalize(s[:,69:73],dim=-1)
        torch.testing.assert_close(features(s,ctx),task.critic_features(s,None,None,ctx),atol=0,rtol=0)

    def test_timeout_bootstraps_terminal_state_but_failure_does_not(self):
        r=torch.tensor([[2.,2.,100.]])
        v=torch.zeros_like(r);nv=torch.full_like(r,10.)
        fail=torch.tensor([[True,False,False]]);timeout=torch.tensor([[True,True,False]])
        valid=torch.tensor([[True,True,False]])
        adv,ret=gae(r,v,nv,fail,timeout,valid,.99,.95)
        torch.testing.assert_close(ret,torch.tensor([[2.,11.9,0.]]))

    def test_rl_games_losses_equal_pinned_rsl_clipping(self):
        torch.manual_seed(6);old=torch.randn(30);new=torch.randn(30,requires_grad=True);adv=torch.randn(30)
        ratio=(new-old).exp();ref=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))
        actual=actor_loss(-old,-new,adv,True,.2);torch.testing.assert_close(actual,ref)
        v=torch.randn(30,requires_grad=True);target=torch.randn(30)
        refv=torch.maximum((v-target).square(),(old+(v-old).clamp(-.2,.2)-target).square())
        torch.testing.assert_close(critic_loss(old,v,.2,target,True),refv)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(),new,retain_graph=True)[0],torch.autograd.grad(ref.sum(),new)[0])

if __name__=='__main__':unittest.main()
