"""Persistent native simulator, without loading a released actor or WSM.

Isaac Gym must be imported before Torch by the worker entry point.
"""
import os
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from isaacgymenvs.tasks.wm_craftnet_camera_rotation import RealmanSharpaHa4CameraRotation
from isaacgymenvs.utils.utils import set_seed
from unionmimic.common import UPSTREAM
from .task import HandTask, S, C
from .curriculum import capture_curriculum, restore_curriculum


class StudyRotation(RealmanSharpaHa4CameraRotation):
    def _update_gravity_curriculum(self, outcomes):
        if getattr(self, 'study_skip_curriculum_update', False):
            return
        return super()._update_gravity_curriculum(outcomes)

    def reset_idx(self, env_ids, goal_env_ids, is_test=False):
        super().reset_idx(env_ids,goal_env_ids,is_test)
        if hasattr(self,'study_properties'):
            for idx in env_ids.tolist():
                e=self.envs[idx]
                body=self.gym.get_actor_rigid_body_properties(e,self.objects[idx])
                shape=self.gym.get_actor_rigid_shape_properties(e,self.objects[idx])
                hand=self.gym.get_actor_rigid_shape_properties(e,self.arm_hands[idx])
                elastomer=next(iter(self.hand_elastomer_shape_ids))
                metal=next(iter(self.hand_other_shape_ids))
                self.study_properties[idx]=torch.tensor([sum(p.mass for p in body),
                    shape[0].friction,hand[elastomer].friction,hand[metal].friction],device=self.device)

    def pre_physics_step(self, actions):
        self.study_command=actions.detach().clone()
        super().pre_physics_step(actions)


def make_simulator(nenv, gpu, seed):
    torch.cuda.set_device(gpu)
    os.chdir(UPSTREAM)
    # Import registers upstream OmegaConf resolvers, but never constructs a player.
    import isaacgymenvs.train  # noqa: F401
    with initialize_config_dir(config_dir=str(UPSTREAM/'isaacgymenvs/cfg'),version_base='1.1'):
        cfg=compose(config_name='config',overrides=[
            'task=WMCraftnetRotation','train=WMCraftnetPPO','headless=True','test=False',
            'wandb_activate=False','capture_video=False','force_render=False',
            'sim_device=cuda:%d'%gpu,'rl_device=cuda:%d'%gpu,'graphics_device_id=%d'%gpu,
            'seed=%d'%seed,'task.env.numEnvs=%d'%nenv,
            'train.params.tuning.num_envs=%d'%nenv,'task.env.objSet=set_z','task.env.axis=z',
            '++task.env.singleObject=set_obj6_block_corner','++task.env.isTestRun=False',
            'task.env.cameraPolicy.enabled=False','task.env.cameraPolicy.worldModel.enabled=False',
            'task.env.cameraDemo.enabled=False','task.env.cameraDemo.inference_video.enabled=False',
            'task.env.wmDepthVideoRecord.enabled=False','task.env.spinTrace.enabled=False',
            'task.env.evalMetrics.enabled=False','++task.env.wmFeatureRecord.enabled=False'])
    set_seed(seed,torch_deterministic=True)
    env=StudyRotation(OmegaConf.to_container(cfg.task,resolve=True),cfg.rl_device,cfg.sim_device,gpu,True,False,False)
    assert env.num_actions==22 and env.n_obs_dim==73 and env.n_stack==4
    assert env.control_mode=='relative' and env.act_moving_average==.8
    assert not env.camera_policy_enabled and env.wm_adapter is None
    assert env.used_training_objects==['set_obj6_block_corner']
    env.study_properties=torch.zeros(nenv,4,device=env.device)
    env.reset_buf[:]=1
    env.step(torch.zeros(nenv,22,device=env.device))
    return env,task_from_env(env),OmegaConf.to_container(cfg,resolve=True)


def task_from_env(env):
    """Read the native reward/controller constants without changing them."""
    mapping=dict(spin_coef='spin_coef',max_spin='reward_max_spin_rate',vel_coef='vel_coef',
        contact_coef='contact_coef',finger_coef='finger_coef',axis_coef='axis_dev_penalty_coef',
        torque_coef='torque_coef',work_coef='work_coef',action_coef='action_penalty_scale',
        control_coef='control_penalty_scale',pose_coef='hand_pose_coef',position_coef='obj_init_pos_dev_coef',
        distance_deadzone='distance_reward_deadzone',distance_width='distance_reward_penalty_width',
        stall_speed='no_spin_angvel_thresh',stall_steps='no_spin_max_steps',stall_penalty='no_spin_reset_penalty',
        fall_dist='fall_dist',fall_penalty='fall_penalty',tip_threshold='tip_force_penalty_thresh',
        tip_coef='tip_force_penalty_coef',tip_alpha='tip_force_penalty_exp_alpha',palm_coef='palm_root_contact_coef',
        thumb_penalty='thumb_non_tip_contact_penalty',thumb_threshold='thumb_non_tip_contact_force_thresh',
        tac_thresh='tac_thresh',episode_steps='max_episode_length',latency='latency',sensor_noise='sensor_noise',
        relative_scale='relative_scale')
    constants={k:float(getattr(env,v)) for k,v in mapping.items()}
    for k,v in dict(lower='arm_hand_dof_lower_limits',upper='arm_hand_dof_upper_limits',
                    default_q='arm_hand_dof_default_pos',pose_weights='hand_pose_joint_weight').items():
        constants[k]=getattr(env,v).tolist()
    constants['control_dt']=env.dt*env.control_freq_inv
    return HandTask(constants).to(env.device)


@torch.no_grad()
def state(env):
    contacts=env.contact_tensor.view(env.num_envs,-1,3).norm(dim=-1)
    tips=contacts[:,env.sensor_handle_indices]*env.sensor_valid_mask
    palm=(contacts[:,env.palm_root_contact_handle_indices]*env.palm_root_contact_valid_mask).max(-1,keepdim=True).values
    thumb=(contacts[:,env.thumb_non_tip_contact_handle_indices]*env.thumb_non_tip_contact_valid_mask).max(-1,keepdim=True).values
    obj=env.root_state_tensor[env.object_indices,:13]
    ext=env.rb_forces[:,env.object_rb_handles,:].reshape(env.num_envs,-1,3).sum(1)
    return torch.cat((env.arm_hand_dof_pos,env.arm_hand_dof_vel,obj,env.fingertip_pos.flatten(1),
        tips,palm,thumb,env.torques,env.dof_vel_finite_diff,ext,env.prev_targets,env.last_actions,
        env.last_contacts,env.no_spin_counter[:,None].float(),env.progress_buf[:,None].float()),-1).clone()


@torch.no_grad()
def context(env):
    p=env.dr_randomizations.get('actions',{})
    assert 'var' in p and 'var_corr' in p,'Only the configured additive Gaussian action noise is implemented'
    corr=p.get('corr',torch.zeros(env.num_envs,22,device=env.device))*p['var_corr']+p['mu_corr']+p['mu']
    return torch.cat((env.object_init_pos,env.object_init_quat,env.goal_pos,env.spin_axis,env.study_properties,
        env.p_gain,env.d_gain,env.gravity_env_vector,env.contact_thresh,
        torch.full((env.num_envs,1),env.action_conservative_scale,device=env.device),corr,
        torch.full((env.num_envs,1),p['var'],device=env.device)),-1).clone()


class Collector:
    def __init__(self, env, task, actor, critic, episode_offset=0):
        self.env,self.task,self.actor,self.critic=env,task,actor,critic
        n=env.num_envs;dev=env.device
        self.episode=torch.arange(n,device=dev)+episode_offset
        self.next_episode=episode_offset+n
        self.frames=0
        self.totals={k:torch.zeros(n,device=dev) for k in ['return','rotation','offaxis','tilt_sum','steps']}
        self.completed=[]
        self.max_reward_error=self.max_action_error=self.max_target_error=0.
        self.parity_rows=0
        self.reset_frame_terminations=0

    @torch.no_grad()
    def step(self, algorithm='sm', noise=.1, log_std=None, normalize=True, actor_version=0):
        env=self.env
        invalid=env.reset_buf.bool().clone()
        if invalid.any():
            ids=invalid.nonzero().flatten();n=len(ids)
            self.episode[ids]=torch.arange(self.next_episode,self.next_episode+n,device=env.device)
            self.next_episode+=n
        s=state(env);ctx=context(env);history=env.obs_buf.detach().clamp(-5,5).clone()
        if normalize:
            self.actor.norm.update(history[~invalid])
            self.critic.norm.update(torch.cat((s[~invalid],ctx[~invalid]),-1))
        mean=self.actor(history)
        if algorithm=='ppo':
            dist=torch.distributions.Normal(mean,log_std.exp().expand_as(mean))
            raw=dist.sample();logp=dist.log_prob(raw).sum(-1)
        else:
            raw=mean+noise*torch.randn_like(mean);logp=torch.zeros(len(mean),device=env.device)
        ci=torch.cat((s,ctx),-1);value=self.critic(ci)
        env.step(raw.clamp(-1,1))
        sn=state(env)
        # Incoming-reset rows are excluded from learning, but a failure during
        # that first new physical frame must still count as an episode failure.
        reward_state=s.clone();reward_ctx=ctx.clone()
        if invalid.any():
            reward_state[invalid,S['object'].start+3:S['object'].start+7]=env.last_object_rot[invalid]
            reward_state[invalid,S['stall']]=0
            reward_ctx[invalid]=context(env)[invalid]
        reward,failure,timeout,_,metrics=self.task.reward(reward_state,sn,reward_ctx)
        valid=~invalid
        if valid.any():
            action,target=self.task.controller(s,env.study_command,env.relative_scale_tensor)
            er=float((reward[valid]-env.rew_buf[valid]).abs().max())
            ea=float((action[valid]-env.actions[valid]).abs().max())
            et=float((target[valid]-env.prev_targets[valid]).abs().max())
            self.max_reward_error=max(self.max_reward_error,er)
            self.max_action_error=max(self.max_action_error,ea)
            self.max_target_error=max(self.max_target_error,et)
            self.parity_rows+=int(valid.sum())
            if er>2e-3 or ea>1e-6 or et>1e-6:
                raise AssertionError('Native parity failed: reward=%g action=%g target=%g'%(er,ea,et))
            assert torch.equal((failure|timeout)[valid],env.reset_buf.bool()[valid]),'Terminal contract mismatch'
        if not all(torch.isfinite(x).all() for x in [s,sn,ctx,reward,history]):
            raise FloatingPointError('Nonfinite simulator transition')
        for k in self.totals:
            self.totals[k][invalid]=0
        self.totals['return']+=env.rew_buf
        for k in ['rotation','offaxis']:
            self.totals[k]+=metrics[k]
        self.totals['tilt_sum']+=metrics['tilt']
        self.totals['steps']+=1
        done=env.reset_buf.bool()
        self.reset_frame_terminations+=int((done&invalid).sum())
        for i in done.nonzero().flatten().tolist():
            row={k:float(v[i]) for k,v in self.totals.items()}
            row.update(episode=int(self.episode[i]),env_id=i,failure=bool(failure[i]),timeout=bool(timeout[i]),
                       actor_version=actor_version,axis_deviation_rad=row['tilt_sum']/max(1,row['steps']))
            self.completed.append(row)
        next_ci=torch.cat((sn,ctx),-1)
        self.frames+=env.num_envs
        return dict(s=s,sn=sn,ctx=ctx,history=history,command=env.study_command.clone(),
                    scale=env.relative_scale_tensor.clone(),raw=raw,mean=mean,logp=logp,
                    value=value,next_value=self.critic(next_ci),reward=env.rew_buf.clone()*.01,
                    failure=failure,timeout=timeout,native_done=done.clone(),valid=valid,episode=self.episode.clone(),
                    version=torch.full_like(self.episode,actor_version))

    def evidence(self):
        return dict(parity_rows=self.parity_rows,max_reward_error=self.max_reward_error,
                    max_action_error=self.max_action_error,max_target_error=self.max_target_error,
                    reset_frame_terminations=self.reset_frame_terminations)

    def restore_curriculum(self, saved):
        """Restore task progress, then start new episodes at the restored level.

        This does not restore PhysX internal state or turn process startup into a
        failed training episode. Domain randomization is redrawn at new resets.
        """
        env = self.env
        restore_curriculum(env, saved)
        env.study_skip_curriculum_update = True
        try:
            env.reset_buf[:] = 1
            env.step(torch.zeros(env.num_envs,22,device=env.device))
        finally:
            env.study_skip_curriculum_update = False
        return capture_curriculum(env)

    def snapshot(self):
        # Diagnostic full controller/physical state. PhysX contact solver caches
        # are not serializable: process resume begins fresh, explicitly new episodes.
        env=self.env
        names=['root_state_tensor','dof_state','obs_buf','last_obs_buf','prev_targets','cur_targets',
               'last_actions','last_contacts','progress_buf','reset_buf','no_spin_counter','rb_forces',
               'p_gain','d_gain','gravity_env_vector','object_init_pos','object_init_quat',
               'contact_thresh','study_properties','gravity_curriculum_outcomes']
        return dict(tensors={k:getattr(env,k).detach().cpu().clone() for k in names},
                    curriculum=capture_curriculum(env),
                    next_episode=self.next_episode,frames=self.frames,
                    resume_semantics='Exact learner/replay/RNG restoration; fresh simulator episodes, no cross-reset transition.',
                    unfinished_episodes={k:v.cpu() for k,v in self.totals.items()})
