import copy
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from unionmimic.online.core import Replay,Actor,Critic,RunningNorm,frozen,lambda_returns,objective,gae
from unionmimic.online.task import spin_delta,qrotate,HandTask,PhysicalModel,S,DS,DC


class OnlineLearning(unittest.TestCase):
    def test_tracking_identity_separates_arms_and_preserves_resume(self):
        from unionmimic.online.tracking import tracking_run_id
        sm=Path('/study/experiment-a/sm/learner')
        ppo=Path('/study/experiment-a/ppo/learner')
        self.assertNotEqual(tracking_run_id(sm,'sm'),tracking_run_id(ppo,'ppo'))
        self.assertEqual(tracking_run_id(sm,'sm'),tracking_run_id(str(sm),'sm'))
        self.assertNotEqual(tracking_run_id(sm,'sm'),tracking_run_id('/study/experiment-b/sm/learner','sm'))
        self.assertNotEqual(tracking_run_id('/study/experiment-a/sm-seed1/learner','sm'),
                            tracking_run_id('/study/experiment-a/sm-seed2/learner','sm'))
        self.assertEqual(tracking_run_id('/study/online-sm-20261102-pilot-v1','sm'),'online-sm-20261102-pilot-v1')

    def test_long_lane_dispatch_and_final_test_share_original_deadline(self):
        import importlib.util
        import hashlib
        import json
        import tempfile
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('lane_test','scripts/online_long_lane.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'study/sm-seed2';clock=[1000.];calls=[];training_calls=[0]
            plan=dict(source_sha256={},acceptance={'engineering_passed':True},
                arms={'sm-seed2':dict(algorithm='sm',seed=92002,gpu=1,hours=12,cycles_per_segment=600)},
                training_config={},development=dict(reserve_seconds=360,envs=16,batches=2,seed=94001),
                locked_test=dict(reserve_seconds=1800,envs=20,batches=10,seed=95001,episodes_each=200))
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            approval=root/'approval.json';approval.write_text(json.dumps(dict(approved=True,
                plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())))
            class Child:
                pid=123
                def __init__(self,cmd,**kwargs):
                    calls.append(cmd)
                    if cmd[1].endswith('train_online.py'):
                        training_calls[0]+=1
                        self_case.assertEqual(cmd[cmd.index('--arm')+1],'sm')
                        self_case.assertEqual(cmd[cmd.index('--seed')+1],'92002')
                        learner=run/'learner';learner.mkdir(exist_ok=True)
                        (learner/'last.pt').write_text('test-checkpoint')
                        state='segment_complete' if training_calls[0]==1 else 'completed'
                        (learner/'status.json').write_text(json.dumps({'status':state}))
                        clock[0]=4000. if training_calls[0]==1 else 42400.1
                    else:
                        dest=Path(cmd[cmd.index('--output')+1]);dest.mkdir(parents=True)
                        n=200 if dest.name=='locked-test' else 32
                        (dest/'evaluation.json').write_text(json.dumps(dict(episodes=[{}]*n,mean_return=3.,mean_rotation=1.,failures=0)))
                        clock[0]+=100.
                def wait(self,timeout=None):return 0
                def poll(self):return 0
            self_case=self
            with patch.object(module,'ROOT',root),patch.object(module.time,'time',side_effect=lambda:clock[0]), \
                 patch.object(module.subprocess,'Popen',Child),patch.object(module.signal,'signal'), \
                 patch('sys.argv',['lane','--plan',str(plan_path),'--approval',str(approval),'--arm','sm-seed2','--run',str(run)]):
                module.main()
            state=json.loads((run/'lane-status.json').read_text())
            self.assertEqual(state['status'],'completed');self.assertEqual(state['deadline_unix'],44200.)
            self.assertFalse(state['locked_test_pending']);self.assertEqual(len(calls),4)
            self.assertIn('--resume',calls[2]);self.assertLess(clock[0],state['deadline_unix'])

    def test_reference_lambda_and_objective_derivatives(self):
        source=Path('work/upstreams/sm-reference-audit/scripts/wm/train_yaw.py').read_text()
        start=source.index('        Rt = V[-1]');end=source.index('        T = torch.stack(tgt[::-1])',start)
        torch.manual_seed(81)
        r=torch.randn(8,9,dtype=torch.float64,requires_grad=True)
        c=torch.rand_like(r,requires_grad=True);v=torch.randn(9,9,dtype=torch.float64)
        scope=dict(V=v,rews=r,conts=c,args=SimpleNamespace(horizon=8,gamma=.99,lam=.95))
        exec(compile(textwrap.dedent(source[start:end]),'pinned-sm-reference','exec'),scope)
        torch.testing.assert_close(lambda_returns(r,c,v),torch.stack(scope['tgt'][::-1]),rtol=0,atol=0)
        weights=torch.cat((torch.ones_like(c[:1]),c.cumprod(0)),0)
        expected=sum(.99**j*weights[j]*r[j] for j in range(8))+.99**8*weights[8]*v[-1]
        actual=objective(r,c,v[-1])
        for a,b in zip(torch.autograd.grad(actual.sum(),(r,c),retain_graph=True),
                       torch.autograd.grad(expected.sum(),(r,c))):torch.testing.assert_close(a,b)

    def test_replay_excludes_reset_and_ring_overwrite(self):
        replay=Replay(8,2,'cpu')
        for i in range(16):
            ep=torch.full((2,),i//4)
            replay.add(dict(episode=ep,valid=torch.full((2,),i%4!=0),
                failure=torch.full((2,),i%4==3),timeout=torch.zeros(2,dtype=torch.bool),
                value=torch.full((2,),i)))
        sample=replay.indices(100,3)
        self.assertTrue(sample['valid'].all())
        self.assertTrue((sample['episode']==sample['episode'][:1]).all())
        self.assertTrue((sample['value'][1:]-sample['value'][:-1]==1).all())
        self.assertTrue((sample['value']>=8).all())
        saved=replay.state_dict();other=Replay(8,2,'cpu');other.load_state_dict(saved)
        torch.manual_seed(9);a=replay.indices(50,3)
        torch.manual_seed(9);b=other.indices(50,3)
        for k in a:torch.testing.assert_close(a[k],b[k])

    def test_timeout_bootstrap_and_reset_separation(self):
        r=torch.tensor([[1.],[2.],[1000.]])
        value=torch.zeros_like(r);nextv=torch.full_like(r,10.)
        fail=torch.tensor([[False],[False],[False]])
        timeout=torch.tensor([[False],[True],[False]])
        valid=torch.tensor([[True],[True],[False]])
        adv,_=gae(r,value,nextv,fail,timeout,valid)
        self.assertAlmostEqual(float(adv[1]),11.9,places=5)
        self.assertEqual(float(adv[2]),0)
        fail[1]=True
        adv,_=gae(r,value,nextv,fail,timeout,valid)
        self.assertEqual(float(adv[1]),2)

    def test_frozen_dynamics_keeps_input_gradient(self):
        model=torch.nn.Linear(1,1,bias=False).double();model.weight.data.fill_(.8)
        theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        before=model.weight.detach().clone()
        def fn(t):
            x=torch.ones(1,1,dtype=torch.float64)*.1;rs=[];cs=[]
            with frozen(model):
                for _ in range(8):
                    x=model(x)+torch.tanh(t*x);rs.append(-x.square().squeeze(-1));cs.append(torch.sigmoid(x.squeeze(-1)))
                return objective(rs,cs,-x.square().squeeze(-1)).sum()
        g=torch.autograd.grad(fn(theta),theta)[0]
        fd=(fn(theta.detach()+1e-5)-fn(theta.detach()-1e-5))/(2e-5)
        torch.testing.assert_close(g,fd,atol=1e-7,rtol=1e-6)
        self.assertIsNone(model.weight.grad);torch.testing.assert_close(before,model.weight)

    def test_rotation_identity_has_finite_derivative(self):
        q=torch.tensor([[0.,0.,0.,1.]],requires_grad=True)
        r,o=spin_delta(q,q,torch.tensor([[0.,0.,1.]]))
        (r+o).sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_learning_checkpoint_optimizer_continuation(self):
        torch.manual_seed(5);a=Actor();opt=torch.optim.Adam(a.parameters(),lr=1e-4)
        x=torch.randn(5,292)
        def step(net,optim):
            optim.zero_grad();loss=(net(x)-.2).square().mean();loss.backward();optim.step()
        step(a,opt);saved=copy.deepcopy((a.state_dict(),opt.state_dict()))
        b=Actor();ob=torch.optim.Adam(b.parameters(),lr=1e-4)
        b.load_state_dict(saved[0]);ob.load_state_dict(saved[1])
        step(a,opt);step(b,ob)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_common_initial_actor_and_no_reference_checkpoint(self):
        torch.manual_seed(192001);a=Actor();ca=Critic(DS+DC)
        torch.manual_seed(192001);b=Actor();cb=Critic(DS+DC)
        for x,y in zip(list(a.state_dict().values())+list(ca.state_dict().values()),
                       list(b.state_dict().values())+list(cb.state_dict().values())):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_complete_sm_update_and_joint_resume_on_physical_replay(self):
        from unionmimic.online.learner import Learner
        ck=torch.load('work/runs/online-sm-20261102-probe02/last.pt',map_location='cpu')
        cfg=ck['config'];cfg.update(batch=8,model_horizon=4,horizon=4,critic_epochs=1)
        def make():
            task=HandTask(ck['task_constants'])
            return Learner(Actor(),Critic(DS+DC),PhysicalModel(task),task,cfg)
        a=make();a.load_state_dict(ck['learner'])
        replay=Replay(ck['replay']['capacity'],ck['replay']['nenv'],'cpu');replay.load_state_dict(ck['replay'])
        a.fit(replay,1)
        frozen_model=copy.deepcopy(a.model.state_dict())
        old_actor=copy.deepcopy(a.actor.state_dict())
        b=make();b.load_state_dict(copy.deepcopy(a.state_dict()))
        rng=torch.get_rng_state();a.imagine(replay)
        torch.set_rng_state(rng);b.imagine(replay)
        for key,value in a.model.state_dict().items():torch.testing.assert_close(value,frozen_model[key],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,old_actor[k]) for k,v in a.actor.state_dict().items()))
        for name in ['actor','critic','model']:
            for key,value in getattr(a,name).state_dict().items():
                torch.testing.assert_close(value,getattr(b,name).state_dict()[key],rtol=0,atol=0)
        self.assertEqual(a.counts,b.counts)

    def test_heldout_episode_split_and_recorded_reward(self):
        ck=torch.load('work/runs/online-sm-20261102-probe02/last.pt',map_location='cpu')
        replay=Replay(ck['replay']['capacity'],ck['replay']['nenv'],'cpu');replay.load_state_dict(ck['replay'])
        train=replay.indices(64,4,split='train');valid=replay.indices(64,4,split='valid')
        self.assertTrue((train['episode']%5!=0).all());self.assertTrue((valid['episode']%5==0).all())
        task=HandTask(ck['task_constants'])
        r,f,t,_,_=task.reward(valid['s'][0],valid['sn'][0],valid['ctx'][0])
        torch.testing.assert_close(r*.01,valid['reward'][0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(f,valid['failure'][0]);torch.testing.assert_close(t,valid['timeout'][0])

    def test_budget_and_stop_prevent_updates(self):
        from unionmimic.online.learner import Learner
        from unionmimic.online.core import BudgetExpired,RunStopped
        obj=object.__new__(Learner);obj.deadline=0;obj.stop_requested=lambda:False
        with self.assertRaises(BudgetExpired):obj.check_budget()
        obj.deadline=float('inf');obj.stop_requested=lambda:True
        with self.assertRaises(RunStopped):obj.check_budget()

    def test_long_launch_requires_matching_confirmation(self):
        import importlib.util,tempfile,json,hashlib
        spec=importlib.util.spec_from_file_location('long_lane','scripts/online_long_lane.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            plan=Path(d)/'plan.json';approval=Path(d)/'approval.json'
            plan.write_text(json.dumps({'source_sha256':{},'acceptance':{'engineering_passed':True}}))
            approval.write_text(json.dumps({'approved':False}))
            with self.assertRaises(PermissionError):mod.verify(plan,approval)
            approval.write_text(json.dumps({'approved':True,'plan_sha256':'incorrect'}))
            with self.assertRaises(PermissionError):mod.verify(plan,approval)
            approval.write_text(json.dumps({'approved':True,'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest()}))
            self.assertTrue(mod.verify(plan,approval)['acceptance']['engineering_passed'])


if __name__=='__main__':unittest.main()
