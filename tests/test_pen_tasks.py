import unittest
from pathlib import Path
import torch
from unionmimic.pen_tasks import PenTasks
from unionmimic.pen_camera import PenCamera

ROOT=Path(__file__).resolve().parents[1]
class PenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.task=PenTasks(ROOT/'work/motions/sharpa-pen-20261104-v1','cpu')
    def test_source_time_and_end(self):
        t=self.task;ids=torch.tensor([2,3]);clips=torch.tensor([0,4]);step=torch.tensor([50,100])
        out=t.target(ids,clips,step)
        torch.testing.assert_close(out['pose'],torch.stack([t.clips[0]['pose'][50],t.clips[4]['pose'][100]]))
        self.assertTrue(t.target(ids,clips,t.lengths[clips])['ended'].all())
    def test_capability_and_reference_goals_differ(self):
        ids=torch.arange(4);clips=torch.tensor([0,0,0,4]);step=torch.full((4,),100)
        g=self.task.target(ids,clips,step)['pose'];self.assertFalse(torch.equal(g[0],g[1]))
        features=self.task.goal_features(ids,clips,step)
        self.assertEqual(features.shape,(4,95))
        torch.testing.assert_close(features[2,:22],self.task.clips[0]['q'][100])
        force=self.task.perturbation(ids,step);self.assertAlmostEqual(float(force[1,1]),.15,places=6)
        self.assertEqual(int(force.count_nonzero()),1)
        self.assertEqual(int(self.task.perturbation(ids,step+10).count_nonzero()),0)
    def test_sensor_matches_age_buffer_and_reset(self):
        n=3;cam=PenCamera(n,'cpu');ids=torch.arange(n);cam.reset(ids)
        cam.bias_pos.zero_();cam.bias_quat.zero_();cam.bias_quat[:,0]=1
        history=[];ages=torch.zeros(n,dtype=torch.long);ticks=torch.zeros(n,dtype=torch.long)
        for step in range(100):
            pose=torch.zeros(n,7);pose[:,0]=step;pose[:,3]=1;history.append(pose)
            roll=torch.tensor([.01,.5,.9]) if step%7 else torch.zeros(n)
            holding=torch.where(ticks>0,(roll<.78)&(ticks<40),roll<.03)
            ticks=torch.where(holding,ticks+1,torch.zeros_like(ticks));aged=ages+1
            ages=torch.where(holding,aged,torch.minimum(torch.ones_like(ages),aged)).clamp(max=step)
            expected=torch.stack([history[step-int(ages[i])][i] for i in range(n)])
            out,cam.previous,cam.held,cam.hold_ticks,cam.ready=cam.transition(pose,cam.bias_pos,cam.bias_quat,
                cam.previous,cam.held,cam.hold_ticks,cam.ready,jitter_pos=torch.zeros(n,3),jitter_quat=cam.bias_quat,roll=roll)
            torch.testing.assert_close(out,expected)
        cam.reset(torch.tensor([1]));self.assertFalse(cam.ready[1]);self.assertTrue(cam.ready[0])
    def test_sensor_restore_and_input_gradient(self):
        a=PenCamera(2,'cpu');a.reset(torch.arange(2));pose=torch.tensor([[0.,0,0,1,0,0,0]]).expand(2,-1).clone().requires_grad_()
        a.advance(pose);b=PenCamera(2,'cpu');b.restore(a.snapshot())
        rng=torch.get_rng_state();x=a.advance(pose+torch.tensor([1.,0,0,0,0,0,0]));torch.set_rng_state(rng)
        y=b.advance(pose+torch.tensor([1.,0,0,0,0,0,0]));torch.testing.assert_close(x,y)
        x[:,:3].sum().backward();self.assertTrue(torch.isfinite(pose.grad).all());self.assertGreater(float(pose.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
