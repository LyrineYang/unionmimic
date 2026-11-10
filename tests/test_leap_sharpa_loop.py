"""Reference identity and adaptation boundaries; tiny CPU tensors are unit fixtures."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa'))
from in_hand_rotation_mjlab.world_model.policy import Actor,Critic
from unionmimic.leap_sharpa.model import SharpaModel,SharpaTask,PhysicalError,X
from native_update import actor_critic_update


class ReferenceAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.constants=json.loads((ROOT/'work/runs/sharpa-leap-operator-20261103-v1/task-constants.json').read_text())

    def fixture(self):
        torch.manual_seed(7)
        m=SharpaModel(self.constants)
        s=torch.zeros(16,177);s[:,:22]=torch.tensor(self.constants['default_q']);s[:,44:66]=s[:,:22]
        s[:,66:69]=torch.tensor([0.,0.,.1]);s[:,69]=1
        s[:,79:94]=s[:,66:69].repeat(1,5);s[:,94:101]=.5
        c=torch.zeros(16,100);c[:,6]=1;c[:,9]=.1;c[:,12]=1;c[:,99]=1
        sn=s.clone();sn[:,:22]+=.001*torch.randn(16,22);sn[:,22:44]=.01*torch.randn(16,22)
        sn[:,66:69]+=.001*torch.randn(16,3);sn[:,73:79]=.01*torch.randn(16,6)
        m.fit_stats(s,sn,c)
        return m,s,sn,c

    def test_reference_physics_policy_replay_are_unmodified(self):
        original=ROOT/'work/upstreams/sm-reference-audit/src/in_hand_rotation_mjlab/world_model'
        generated=ROOT/'work/runtime/leap-sharpa/src/in_hand_rotation_mjlab/world_model'
        for name in ('policy.py','replay.py'):
            self.assertEqual((original/name).read_bytes(),(generated/name).read_bytes())
        self.assertEqual((original/'model.py').read_text().replace('from .record import state_slices','from .layout import state_slices'),(generated/'model.py').read_text())

    def test_original_lambda_returns_and_optimizer_loop_preserved(self):
        source=(ROOT/'work/upstreams/sm-reference-audit/scripts/wm/train_yaw.py').read_text()
        generated=(ROOT/'work/runtime/leap-sharpa/native_update.py').read_text()
        # Compare executed reference statements, ignoring only added finite assertions.
        def block(text):
            a=text.index('with torch.no_grad():',text.index('# 3. critic:'))
            b=text.index('st.update(J=',a)
            lines=text[a:b].splitlines();indent=len(lines[1])-len(lines[1].lstrip())-2
            code=lines[0]+'\n'+'\n'.join(x[indent:] if x.startswith(' '*indent) else x for x in lines[1:])
            tree=ast.parse(code)
            class RemoveAssertions(ast.NodeTransformer):
                def visit_Assert(self,node):return None
            return ast.dump(RemoveAssertions().visit(tree),include_attributes=False)
        self.assertEqual(block(source),block(generated))

    def test_actor_update_never_reads_future_supervision(self):
        import inspect
        tree=ast.parse(inspect.getsource(actor_critic_update))
        for n in ast.walk(tree):
            if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='replay':
                self.assertNotIn(n.attr,('s_next','rew','done','failure'))

    def test_failure_supervision_does_not_update_physics_or_sensors(self):
        model,s,sn,c=self.fixture();nc=model.normalize(c)
        with torch.no_grad():prediction,_=model(s,torch.zeros(16,22),nc,scale=torch.full((16,1),.2))
        torch.nn.functional.binary_cross_entropy_with_logits(model.failure_logit(prediction,nc),torch.ones(16)).backward()
        self.assertTrue(all(p.grad is None for p in model.physics.parameters()))
        self.assertTrue(all(p.grad is None for p in model.sensors.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.term.parameters()))

    def test_frozen_model_allows_action_gradient_and_predicted_observations(self):
        model,s,sn,c=self.fixture();task=SharpaTask(model,self.constants,'cpu');nc=model.normalize(c)
        before=copy.deepcopy(model.state_dict())
        for p in model.parameters():p.requires_grad_(False)
        command=torch.full((16,22),.2,requires_grad=True)
        prediction,_=model(s,command,nc,scale=torch.full((16,1),.2))
        frame,updated=task.imagined_frame(prediction,nc)
        self.assertEqual(frame.shape,(16,73));self.assertEqual(updated.shape,(16,177))
        frame[:,:44].sum().backward()
        self.assertTrue(torch.isfinite(command.grad).all());self.assertGreater(float(command.grad.abs().sum()),0)
        for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)

    def test_native_step_reward_units_and_bound(self):
        model,s,sn,c=self.fixture();task=SharpaTask(model,self.constants,'cpu');nc=model.normalize(c)
        from unionmimic.leap_sharpa.model import legacy_state
        r,*_=task.native.reward(legacy_state(s,c),legacy_state(sn,c),c[:,:93])
        terms,_=task.reward_rates(s,sn,None,None,nc)
        torch.testing.assert_close(terms['native']*self.constants['control_dt'],r)
        self.assertAlmostEqual(task.max_step_reward(),3.5)

    def test_exogenous_force_decay_and_recorded_force_override(self):
        model,s,sn,c=self.fixture();c[:,13]=.05
        model.constants=dict(model.constants,force_scale=.5,force_probability=0.,force_decay=.99)
        s[:,X['external_force']]=torch.tensor([.1,.2,.3]);nc=model.normalize(c)
        torch.testing.assert_close(model.next_force(s,nc),s[:,X['external_force']]*.99)
        actual=torch.full((16,3),.02)
        predicted,_=model(s,torch.zeros(16,22),nc,scale=torch.full((16,1),.2),external_force=actual)
        torch.testing.assert_close(predicted[:,X['external_force']],actual)

    def test_physical_fit_path_preserves_eight_step_loss_and_gradients(self):
        model,s,sn,c=self.fixture()
        original=copy.deepcopy(model)
        command=torch.full((16,22),.2)
        inputs=dict(scale=torch.full((16,1),.2),external_force=torch.full((16,3),.02))
        old_state=new_state=s
        old_loss=new_loss=0.
        error=PhysicalError()
        error.fit(s,sn)
        for _ in range(8):
            old_state,_=original(old_state,command,original.normalize(c),**inputs)
            new_state=model.physical_transition(new_state,command,model.normalize(c),**inputs)
            torch.testing.assert_close(old_state[:,:79],new_state[:,:79],atol=0,rtol=0)
            torch.testing.assert_close(old_state[:,145:],new_state[:,145:],atol=0,rtol=0)
            old_loss+=error(old_state,sn).mean()
            new_loss+=error(new_state,sn).mean()
        torch.testing.assert_close(original.failure_logit(old_state,original.normalize(c)),
                                   model.failure_logit(new_state,model.normalize(c)),atol=0,rtol=0)
        old_loss.backward()
        new_loss.backward()
        for old,new in zip(original.physics.parameters(),model.physics.parameters()):
            torch.testing.assert_close(old.grad,new.grad)
        self.assertTrue(all(p.grad is None for p in model.sensors.parameters()))

    def test_zero_contact_labels_keep_force_loss_and_gradient_finite(self):
        model,s,sn,c=self.fixture();sn[:,94:101]=0
        model.fit_stats(s,sn,c)
        loss=model.sensor_loss(s,sn,model.normalize(c));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.sensors.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
