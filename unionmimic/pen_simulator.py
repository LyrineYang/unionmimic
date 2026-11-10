"""Isolated Sharpa pen environment. Does not alter active rotation sources."""
import os,json
from pathlib import Path
import numpy as np
import torch,trimesh
from isaacgym import gymapi,gymtorch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from unionmimic.common import UPSTREAM
from unionmimic.online_baseline.simulator import StudyRotation
from unionmimic.online_baseline.task import qrotate,qconj
from unionmimic.motions import qmul,qrot,qinv
from unionmimic.pen_tasks import PenTasks,ResidualCommand,LINK_NAMES
from unionmimic.pen_camera import PenCamera
from unionmimic.pen_reward import reward_terms,update_charge
from unionmimic.pen_sampling import ReferenceSampler
ROOT=Path(__file__).resolve().parents[1]

class PenEnvironment(StudyRotation):
    def create_object_asset_dict(self,asset_root):
        self.used_training_objects=['reference_pen'];self.num_training_objects=1
        options=gymapi.AssetOptions();options.density=1
        obj=self.gym.load_asset(self.sim,str(ROOT/'assets/reference_pen'),'pen_tracking220.urdf',options)
        options.disable_gravity=True;goal=self.gym.load_asset(self.sim,str(ROOT/'assets/reference_pen'),'pen_tracking220.urdf',options)
        self.object_asset_dict={'reference_pen':{'obj':obj,'goal':goal}}
        self.object_rb_count=self.gym.get_asset_rigid_body_count(obj)
        mesh=trimesh.creation.cylinder(radius=.00745,height=.148,sections=32)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
        caps=[]
        for sign in (-1,1):
            cap=trimesh.creation.box(extents=[.036,.018,.018]);cap.apply_translation([sign*.092,0,0]);caps.append(cap)
        mesh=trimesh.util.concatenate([mesh]+caps)
        self._object_bps_by_name['reference_pen']=self._compute_object_bps_vector(mesh)

    def local_pose(self):
        base=self.hand_start_states
        obj=self.root_state_tensor[self.object_indices,:7]
        bq=base[:,[6,3,4,5]];oq=obj[:,[6,3,4,5]]
        return torch.cat((qrot(qinv(bq),obj[:,:3]-base[:,:3]),qmul(qinv(bq),oq)),1)

    def reset_idx(self,ids,goal_ids,is_test=False):
        if not hasattr(self,'pen_tasks'):return super().reset_idx(ids,goal_ids,is_test)
        if not len(ids):return
        if hasattr(self,'pen_failure'):
            scored=ids[(self.task_ids[ids]>=2)&(self.progress_buf[ids]>0)]
            self.reference_sampler.record(self.clip_ids[scored],self.reference_start_step[scored],
                self.reference_step[scored],self.pen_failure[scored])
        # One state injection per simulator step. Do not first reset to the old
        # corner-block posture and then write a second incompatible root/DOF state.
        self._sample_gravity_for_resets(ids)
        if self.randomize:self.apply_randomizations(self.randomization_params)
        for i in ids.tolist():
            self._randomize_hand_object_friction(self.envs[i],self.arm_hands[i],self.objects[i])
        if not getattr(self,'fixed_clips',False):
            self.clip_ids[ids]=self.pen_tasks.choose(self.task_ids[ids])
        if getattr(self,'fixed_phase',None) is not None:
            self.reference_step[ids]=self.fixed_phase
        else:
            # Reference-state initialization. Hold branch uses zero velocities,
            # moving branch carries the sampled trajectory's velocities.
            self.reference_step[ids]=0
            for task,pool in ((2,self.pen_tasks.single),(3,self.pen_tasks.multi)):
                selected=ids[self.task_ids[ids]==task]
                if len(selected):
                    clips,phase=self.reference_sampler.sample(pool,len(selected))
                    self.clip_ids[selected]=clips;self.reference_step[selected]=phase
        self.reference_start_step[ids]=self.reference_step[ids]
        ref=self.pen_tasks.target(self.task_ids[ids],self.clip_ids[ids],self.reference_step[ids])
        q=ref['q'];pose=ref['pose']
        base=self.hand_start_states[ids];bq=base[:,[6,3,4,5]]
        worldq=qmul(bq,pose[:,3:]);worldp=qrot(bq,pose[:,:3])+base[:,:3]
        oi=self.object_indices[ids];self.root_state_tensor[oi,:3]=worldp
        self.root_state_tensor[oi,3:7]=worldq[:,[1,2,3,0]]
        self.root_state_tensor[oi,7:10]=qrot(bq,ref['v']);self.root_state_tensor[oi,10:13]=qrot(bq,ref['w'])
        self.arm_hand_dof_pos[ids]=q;self.arm_hand_dof_vel[ids]=ref['qd']
        self.prev_targets[ids]=q;self.cur_targets[ids]=q;self.last_actions[ids]=0
        self.object_init_pos[ids]=worldp;self.object_init_quat[ids]=worldq[:,[1,2,3,0]]
        if hasattr(self,'goal_pos'):self.goal_pos[ids]=worldp
        self.goal_states[ids,:3]=worldp
        self.progress_buf[ids]=0;self.reset_buf[ids]=0;self.reset_goal_buf[ids]=0;self.successes[ids]=0;self.no_spin_counter[ids]=0
        self.obs_buf[ids]=0;self.last_obs_buf[ids]=0;self.init_stack_buf[ids]=1;self.last_contacts[ids]=0
        self.contact_thresh[ids]=(torch.rand_like(self.contact_thresh[ids])*2-1)*self.tac_thresh_rand+self.tac_thresh
        self.rb_forces[ids]=0;self.raw_previous[ids]=0;self.raw_previous_previous[ids]=0;self.current_charge[ids]=0
        self.p_gain[ids]=self.p_gain_defaults.unsqueeze(0).repeat(len(ids),1);self.d_gain[ids]=self.d_gain_defaults.unsqueeze(0).repeat(len(ids),1)
        if self.pd_dr_enabled and self.randomize and not is_test:
            span=self.pd_dr_scale_upper-self.pd_dr_scale_lower
            self.p_gain[ids]*=self.pd_dr_scale_lower+torch.rand(len(ids),22,device=self.device)*span
            self.d_gain[ids]*=self.pd_dr_scale_lower+torch.rand(len(ids),22,device=self.device)*span
        self.spin_axis[ids]=torch.tensor([0.,0.,1.],device=self.device).expand(len(ids),-1).contiguous()
        oi=oi.to(torch.int32);hi=self.hand_indices[ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),gymtorch.unwrap_tensor(oi),len(ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(hi),len(ids))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.prev_targets),gymtorch.unwrap_tensor(hi),len(ids))
        self.pose_sensor.reset(ids,pose)
        if hasattr(self,'study_properties'):
            for i in ids.tolist():
                e=self.envs[i];body=self.gym.get_actor_rigid_body_properties(e,self.objects[i]);shape=self.gym.get_actor_rigid_shape_properties(e,self.objects[i])
                hand=self.gym.get_actor_rigid_shape_properties(e,self.arm_hands[i])
                self.study_properties[i]=torch.tensor([sum(p.mass for p in body),shape[0].friction,
                    hand[next(iter(self.hand_elastomer_shape_ids))].friction,hand[next(iter(self.hand_other_shape_ids))].friction],device=self.device)

    def pre_physics_step(self,actions):
        if not hasattr(self,'pen_tasks'):return super().pre_physics_step(actions)
        ids=self.reset_buf.nonzero().flatten();self.reset_idx(ids,self.reset_goal_buf.nonzero().flatten())
        self.study_command=actions.clone();self.last_object_rot=self.root_state_tensor[self.object_indices,3:7].clone()
        ref=self.pen_tasks.target(self.task_ids,self.clip_ids,self.reference_step)
        target,residual=self.pen_controller(ref['q'],actions,self.last_actions)
        self.transition_previous=self.raw_previous.clone();self.transition_older=self.raw_previous_previous.clone()
        self.raw_previous_previous=self.raw_previous.clone();self.raw_previous=actions.clone()
        self.cur_targets[:]=target;self.prev_targets[:]=target;self.last_actions=residual;self.actions=actions.clone()
        self.relative_scale_tensor.fill_(.5)  # stored command scale; controller is reference-residual, not accumulated.
        pulse=self.pen_tasks.perturbation(self.task_ids,self.reference_step)
        self.rb_forces.zero_();self.rb_forces[:,self.object_rb_handles[0]]=pulse
        self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(self.rb_forces),None,gymapi.LOCAL_SPACE)

    def compute_reward(self,actions):
        if not hasattr(self,'pen_tasks'):
            self.rew_buf.zero_();self.reset_buf.zero_();return
        self.current_charge=update_charge(self.current_charge,self.torques,self.arm_hand_dof_effort_limits)
        terms,f,end,m=reward_terms(self.pen_tasks,self.local_pose(),self.arm_hand_dof_pos,self.torques,self.current_charge,
            self.raw_previous,self.transition_previous,self.transition_older,self.task_ids,self.clip_ids,self.reference_step,
            self.arm_hand_dof_lower_limits,self.arm_hand_dof_upper_limits,self.arm_hand_dof_effort_limits)
        self.pen_reward_terms=terms
        self.rew_buf[:]=sum(terms.values())*.02;self.reset_buf[:]=(f|end).long();self.pen_metrics=m
        self.pen_failure=f;self.pen_timeout=end

    def step(self,actions):
        result=super().step(actions)
        if hasattr(self,'pen_timeout'):
            self.timeout_buf=self.pen_timeout & ~self.pen_failure
            self.extras['time_outs']=self.timeout_buf.to(self.rl_device)
        return result

    def post_physics_step(self):
        if hasattr(self,'reference_step'):self.reference_step+=1
        if hasattr(self,'reference_sampler'):self.reference_sampler.update()
        super().post_physics_step()
        if hasattr(self,'pose_sensor'):
            pose=self.pose_sensor.advance(self.local_pose())
            goals=self.pen_tasks.goal_features(self.task_ids,self.clip_ids,self.reference_step)
            self.policy_frame=torch.cat((self.obs_buf[:,:73],self.pen_tasks.endpoints(pose,bounded=True),goals),1)

def make_pen_simulator(nenv,gpu,seed,retarget,render=False,fixed_phase=None,physics_numerics=None):
    retarget=Path(retarget).resolve()
    torch.cuda.set_device(gpu);os.chdir(UPSTREAM)
    tasks=PenTasks(retarget,'cuda:%d'%gpu)
    import isaacgymenvs.train
    from isaacgymenvs.utils.utils import set_seed
    with initialize_config_dir(config_dir=str(UPSTREAM/'isaacgymenvs/cfg'),version_base='1.1'):
        cfg=compose(config_name='config',overrides=[
            'task=WMCraftnetRotation','train=WMCraftnetPPO','headless=True','test=False','wandb_activate=False',
            'capture_video=False','force_render=False','sim_device=cuda:%d'%gpu,'rl_device=cuda:%d'%gpu,
            'graphics_device_id=%d'%gpu,'seed=%d'%seed,'task.env.numEnvs=%d'%nenv,'train.params.tuning.num_envs=%d'%nenv,
            'task.env.objSet=set_z','task.env.axis=z','++task.env.singleObject=set_obj6_block_corner','++task.env.isTestRun=False',
            'task.env.cameraPolicy.enabled=False','task.env.cameraPolicy.worldModel.enabled=False','task.env.cameraDemo.enabled=False',
            'task.env.cameraDemo.inference_video.enabled=False','task.env.wmDepthVideoRecord.enabled=False','task.env.spinTrace.enabled=False',
            'task.env.evalMetrics.enabled=False','++task.env.wmFeatureRecord.enabled=False',
            'task.sim.dt=0.005','task.env.controlFrequencyInv=4','task.env.m_low=0.03','task.env.m_up=0.03','task.env.forceScale=0.0'])
    set_seed(seed,torch_deterministic=True);raw=OmegaConf.to_container(cfg.task,resolve=True)
    raw['enableCameraSensors']=render
    raw['env']['relScale']=.5
    raw['env']['actionsMovingAverage']=.5
    if physics_numerics:
        options=dict(physics_numerics)
        fixed_friction=options.pop('fixed_friction_scale',None)
        assert set(options)<= {'max_depenetration_velocity','num_velocity_iterations'}
        raw['sim']['physx'].update(options)
        if fixed_friction is not None:
            friction=raw['env']['frictionRandomization']
            lower,upper=friction['scale_range']
            assert lower<=fixed_friction<=upper
            friction['scale_range']=[fixed_friction,fixed_friction]
    if tasks.manifest.get('pen_limits')=='hardware_90pct':
        raw['env']['trainLimit']={'global_scale':.9,'unilateral_override_deg':{}}
    if 'gravityCurriculum' in raw['env']:raw['env']['gravityCurriculum']['enabled']=False
    env=PenEnvironment(raw,cfg.rl_device,cfg.sim_device,gpu,True,False,False)
    # Native code assumes the hand is actor/body zero; the white ground precedes
    # it. Resolve object bodies through Gym, never infer from the hand body count.
    old_handles=env.object_rb_handles.clone()
    body_names=env.gym.get_actor_rigid_body_names(env.envs[0],env.objects[0])
    handles=[env.gym.find_actor_rigid_body_index(env.envs[0],env.objects[0],name,gymapi.DOMAIN_ENV) for name in body_names]
    env.object_rb_handles=torch.tensor(handles,device=env.device,dtype=torch.long)
    env.body_index_audit=dict(old=old_handles.tolist(),resolved=handles,names=body_names)
    env.pen_tasks=tasks;env.task_ids=torch.arange(nenv,device=env.device)%4
    env.reference_step=torch.zeros(nenv,dtype=torch.long,device=env.device);env.fixed_phase=fixed_phase
    env.reference_start_step=env.reference_step.clone();env.reference_sampler=ReferenceSampler(tasks.lengths)
    env.raw_previous=torch.zeros(nenv,22,device=env.device);env.raw_previous_previous=torch.zeros_like(env.raw_previous);env.current_charge=torch.zeros_like(env.raw_previous)
    env.transition_previous=torch.zeros_like(env.raw_previous);env.transition_older=torch.zeros_like(env.raw_previous)
    env.pen_controller=ResidualCommand(env.arm_hand_dof_lower_limits,env.arm_hand_dof_upper_limits).to(env.device)
    assert list(env.arm_hand_dof_names)==tasks.manifest['joint_names'],'Retarget joint order mismatch'
    assert (tasks.initial_q>=env.arm_hand_dof_lower_limits-1e-6).all() and (tasks.initial_q<=env.arm_hand_dof_upper_limits+1e-6).all()
    env.clip_ids=env.pen_tasks.choose(env.task_ids);env.pose_sensor=PenCamera(nenv,env.device)
    env.study_properties=torch.zeros(nenv,4,device=env.device)
    env.reset_buf[:]=1;env.step(torch.zeros(nenv,22,device=env.device))
    assert abs(env.dt*env.control_freq_inv-.02)<1e-9
    assert env.policy_frame.shape==(nenv,174) and env.used_training_objects==['reference_pen']
    return env,raw
