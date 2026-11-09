import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/runtime/leap-sharpa/src'))
import torch
from unionmimic.pen_tasks import PenTasks,ResidualCommand,pose_velocity
from unionmimic.pen_reward import reward_terms,update_charge
from unionmimic.leap_sharpa.pen_model import PenModel,PenTask,DX,DC

class PenLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks=PenTasks(ROOT/'work/motions/sharpa-pen-20261104-v1','cpu')
        cls.cfg=json.loads((ROOT/'work/runs/sharpa-leap-12h-20261103-v1/task-constants.json').read_text())
        cls.cfg.update(effort=[1.]*22,control_dt=.02)
    def test_reference_velocity_and_hold(self):
        t=self.tasks;ids=torch.tensor([2,2]);clip=torch.tensor([0,0]);step=torch.tensor([-1,10])
        r=t.target(ids,clip,step)
        self.assertEqual(float(r['qd'][0].abs().sum()+r['v'][0].abs().sum()+r['w'][0].abs().sum()),0.)
        torch.testing.assert_close(r['qd'][1],(t.clips[0]['q'][11]-t.clips[0]['q'][10])/.02)
        p=torch.zeros(3,7);p[:,0]=torch.tensor([0.,.02,.04]);theta=torch.tensor([0.,.02,.04]);p[:,3]=torch.cos(theta/2);p[:,6]=torch.sin(theta/2)
        v,w=pose_velocity(p);torch.testing.assert_close(v[:,0],torch.ones(3));torch.testing.assert_close(w[:,2],torch.ones(3))
    def test_residual_is_reference_based_not_integrated(self):
        f=ResidualCommand(torch.full((22,),-2.),torch.full((22,),2.));q=torch.ones(1,22)*.3
        first,res=f(q,torch.ones(1,22),torch.zeros(1,22));second,res2=f(q,torch.zeros(1,22),res)
        torch.testing.assert_close(first,torch.ones_like(q)*.55);torch.testing.assert_close(second,torch.ones_like(q)*.425)
        clipped,res=f(q,torch.ones(1,22)*10,torch.zeros_like(q));torch.testing.assert_close(res,torch.ones_like(q)*.3)
    def test_reward_units_and_action_second_difference(self):
        t=self.tasks;ids=torch.tensor([2]);clip=torch.tensor([0]);step=torch.tensor([0]);r=t.target(ids,clip,step)
        z=torch.zeros(1,22);low=torch.tensor(self.cfg['lower']);high=torch.tensor(self.cfg['upper']);effort=torch.ones(22)
        terms,f,end,m=reward_terms(t,r['pose'],r['q'],z,z,z,z,z,ids,clip,step,low,high,effort)
        self.assertAlmostEqual(float(sum(terms.values())*.02),.24,places=6)
        a=z.clone();a[0,0]=1
        terms,*_=reward_terms(t,r['pose'],r['q'],z,z,a,z,z,ids,clip,step,low,high,effort)
        self.assertAlmostEqual(float(terms['action_rate']),-.2,places=6)
        c=z
        for _ in range(300):c=update_charge(c,torch.ones_like(z),effort)
        self.assertTrue((c==1).all());self.assertFalse(f.any());self.assertFalse(end.any())
    def test_imagination_advances_memory_with_frozen_model(self):
        torch.manual_seed(1);m=PenModel(self.cfg,self.tasks);task=PenTask(m,self.cfg,'cpu');m.requires_grad_(False)
        s=torch.zeros(4,DX);s[:,69]=1;s[:,180]=1;s[:,187]=1;s[:,192]=1
        c=torch.zeros(4,DC);c[:,96]=1;c[:,105]=1;c[:,100]=torch.arange(4);c[:,101]=torch.tensor([0,0,0,4])
        a=torch.full((4,22),.1,requires_grad=True);n,_=m(s,a,c);frame,n=task.imagined_frame(n,c)
        self.assertEqual(frame.shape,(4,174));self.assertEqual(task.critic_features(n,None,None,c).shape,(4,465))
        torch.testing.assert_close(n[:,176],torch.ones(4));torch.testing.assert_close(n[:,193:215],a)
        torch.testing.assert_close(frame[:,73:79],self.tasks.endpoints(s[:,177:184],bounded=True))
        frame.sum().backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertGreater(float(a.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in m.parameters()))

if __name__=='__main__':unittest.main()
