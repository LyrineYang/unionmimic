import unittest
from pathlib import Path
import torch
from unionmimic.motions import MotionLibrary,ReferenceClock,wrist_local,qmul,qrot,qslerp,tracking_metrics
ROOT=Path(__file__).resolve().parents[1]

class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.lib=MotionLibrary(ROOT/'work/reference-audits/pen-motion-20261103/clips')
    def test_catalog_hashes_and_counts(self):
        self.assertEqual(len(self.lib.clips),13)
        self.assertEqual(sum(c['group']==self.lib.GROUPS[0] for c in self.lib.clips),4)
        self.assertAlmostEqual(sum(c['duration'] for c in self.lib.clips),246.76,places=3)
    def test_wrist_local_cancels_world_transform(self):
        q=torch.tensor([[.70710678,0.,0.,.70710678]])
        p=torch.tensor([[1.,2.,3.]]);local=torch.tensor([[.02,-.03,.1]])
        obj=torch.cat((p+qrot(q,local),q),-1);wrist=torch.cat((p,q),-1)
        pose,links=wrist_local(wrist,obj,(p+qrot(q,local))[:,None])
        torch.testing.assert_close(pose[:,:3],local);torch.testing.assert_close(links[:,0],local)
        torch.testing.assert_close(pose[:,3:],torch.tensor([[1.,0.,0.,0.]]))
    def test_real_and_imaginary_clock_restore_match_and_no_slowdown(self):
        ids=torch.tensor([0,4]);clock=ReferenceClock(self.lib,ids,torch.zeros(2),.02)
        for _ in range(17):clock.advance()
        restored=ReferenceClock(self.lib,**clock.snapshot())
        torch.testing.assert_close(clock.target(16)['pose'],restored.target(16)['pose'])
        torch.testing.assert_close(clock.target()['pose'],self.lib.sample(ids,torch.full((2,),.34))['pose'])
        with self.assertRaises(ValueError):self.lib.validate_environment(control_dt=.10002,has_sharpa_retarget=True,has_pen_contact_initialization=True)
        with self.assertRaises(ValueError):self.lib.validate_environment(control_dt=.02,has_sharpa_retarget=False,has_pen_contact_initialization=True)
    def test_clip_end_does_not_wrap_to_another_clip(self):
        ids=torch.arange(13);x=self.lib.sample(ids,self.lib.ends+1)
        self.assertTrue(x['ended'].all())
        torch.testing.assert_close(x['pose'],torch.stack([c['pose'][-1] for c in self.lib.clips]))
    def test_antipodal_quaternions_and_failed_tracking(self):
        q=torch.tensor([[1.,0.,0.,0.]])
        torch.testing.assert_close(qslerp(q,-q,torch.tensor([[.5]])),q)
        target=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
        actual=target.clone();actual[:,0]=.051
        self.assertTrue(tracking_metrics(actual,target)['failure'].item())
        actual=target.clone();actual[:,3:]=torch.tensor([[0.,0.,1.,0.]])
        self.assertTrue(tracking_metrics(actual,target)['failure'].item())
        actual[:,3:]=torch.tensor([[0.,1.,0.,0.]])
        self.assertFalse(tracking_metrics(actual,target)['failure'].item())

if __name__=='__main__':unittest.main()
