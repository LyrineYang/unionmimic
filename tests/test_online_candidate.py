import copy
import unittest
import torch
from unionmimic.online_candidate.task import HandTask,PhysicalModel,DS,DC
from unionmimic.online_candidate.core import Actor,Critic,Replay,frozen
from unionmimic.online_candidate.learner import Learner
from unionmimic.online.task import PhysicalModel as OriginalModel

class CandidateContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ck=torch.load('work/runs/online-sm-20261102-probe02/last.pt',map_location='cpu')
    def make(self):
        ck=self.ck;task=HandTask(ck['task_constants']);cfg=dict(ck['config'],batch=8,model_horizon=4,horizon=4,critic_epochs=1)
        obj=Learner(Actor(),Critic(DS+DC),PhysicalModel(task),task,cfg);obj.load_state_dict(copy.deepcopy(ck['learner']))
        r=Replay(ck['replay']['capacity'],ck['replay']['nenv'],'cpu');r.load_state_dict(ck['replay'])
        return obj,r
    def test_forward_and_checkpoint_compatibility(self):
        obj,r=self.make();old=OriginalModel(obj.task);old.load_state_dict(obj.model.state_dict())
        b=r.indices(16);inputs=[b[k][0] for k in ['s','command','ctx','scale']]
        for a,b in zip(obj.model(*inputs),old(*inputs)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_terminal_loss_does_not_update_physics_but_input_gradient_survives(self):
        obj,r=self.make();b=r.indices(16)
        with torch.no_grad():sn,_=obj.model(b['s'][0],b['command'][0],b['ctx'][0],b['scale'][0])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(obj.model.termination_logit(sn,b['ctx'][0]),b['failure'][0].float())
        loss.backward()
        self.assertTrue(all(p.grad is None for p in obj.model.net.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in obj.model.term.parameters()))
        sn=sn.detach().requires_grad_(True)
        with frozen(obj.model):g=torch.autograd.grad(obj.model.termination_logit(sn,b['ctx'][0]).sum(),sn)[0]
        self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
    def test_fit_actor_update_and_exact_resume(self):
        a,r=self.make();initial=copy.deepcopy(a.actor.state_dict());physics=copy.deepcopy(a.model.state_dict())
        a.fit(r,2)
        self.assertTrue(any(not torch.equal(v,physics[k]) for k,v in a.model.state_dict().items()))
        b,_=self.make();b.load_state_dict(copy.deepcopy(a.state_dict()));frozen_state=copy.deepcopy(a.model.state_dict())
        rng=torch.get_rng_state();a.imagine(r);torch.set_rng_state(rng);b.imagine(r)
        for k,v in a.model.state_dict().items():torch.testing.assert_close(v,frozen_state[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in a.actor.state_dict().items()))
        for mod in ['model','actor','critic']:
            for k,v in getattr(a,mod).state_dict().items():torch.testing.assert_close(v,getattr(b,mod).state_dict()[k],rtol=0,atol=0)
        self.assertEqual(a.counts,b.counts)

if __name__=='__main__':unittest.main()
