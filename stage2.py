import os
import math
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float, quat_conjugate, quat_mul
from glob import glob
from utils.misc import tprint
from .base.vec_task import VecTask


class AllegroHandCube(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(config['env']['object'])
        # 4. setup reward
        self._setup_reward_config(config['env']['reward'])
        self.base_obj_scale = config['env']['baseObjScale']
        self.save_init_pose = config['env']['genGrasps']
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.evaluate = self.config['on_evaluation']
        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.debug_viz = self.config['env']['enableDebugVis']
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # object apply random forces parameters
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # if self.randomize_scale and self.scale_list_init:
        #     self.saved_grasping_states = {}
        #     for s in self.randomize_scale_list:
        #         self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(
        #             f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy'
        #         )).float().to(self.device)
        # else:
        #     assert self.save_init_pose
        self.saved_grasping_states = {}
        self.saved_grasping_states["1.0"] = torch.from_numpy(np.load(
            f'/home/ubuntu/cube/cache/public_allegro_grasp_50k_s1_2contacts.npy'
        )).float().to(self.device)

        # Also override scale list to avoid KeyError during reset
        self.randomize_scale_list = [1.0]

        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # useful buffers
        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain

        self.object_angvel_prev = self.object_angvel.clone()
        self.cum_yaw_buf = torch.zeros((self.num_envs,),
                                       dtype = torch.float,
                                       device = self.device)
        self.ep_has_succeeded = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.SUCCESS_ANGLE_RAD = 2 * math.pi

        # 每个 env 的本局角速度和/步数（用于计算每局平均角速度）
        self.ep_angvel_sum = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.ep_step_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # 全局累计（跨多局）：成功率/平均角速度
        self.total_episodes = 0
        self.total_successes = 0
        self.sum_episode_mean_angvel = 0.0  # 累加“每局平均角速度”的和

        # 指标输出文件（CSV），默认写到 outputs/paper_metrics/metrics.csv
        out_dir = os.path.join("outputs", "paper_metrics")
        os.makedirs(out_dir, exist_ok=True)
        self.metrics_fpath = os.path.join(out_dir, "metrics.csv")
        # 若首次写入则加表头
        if not os.path.exists(self.metrics_fpath):
            with open(self.metrics_fpath, "w") as f:
                f.write("episodes,avg_angvel_rad_s,avg_success\n")

        self.success_spin_time = self.config['env'].get('successSpinTime', 2)   # 成功所需连续时长(秒)
        self.success_omega_thresh = self.config['env'].get('successOmegaThresh', float("-inf"))  # 判定“在转”的角速度阈值(rad/s)
        self.require_sign = self.config['env'].get('successRequireDirection', False)    # 是否要求固定方向（True=只认一侧）

        # 连续转动累计时间（每个 env 一条）
        self.spin_time_buf = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # 存每局的 ω·(-ẑ) 累加和
        self.ep_w_target_sum = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        # 存每局累积的 step 数
        self.ep_w_target_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)


        # debug and understanding statistics
        self.env_timeout_counter = to_torch(np.zeros((len(self.envs)))).long().to(self.device)  # max 10 (10000 envs)
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()

        # set allegro_hand dof properties
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            allegro_hand_dof_props['effort'][i] = 0.5
            if self.torque_control:
                allegro_hand_dof_props['stiffness'][i] = 0.
                allegro_hand_dof_props['damping'][i] = 0.
                allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                allegro_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                allegro_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            allegro_hand_dof_props['friction'][i] = 0.01
            allegro_hand_dof_props['armature'][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)

        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_allegro_hand_bodies + 2
        max_agg_shapes = self.num_allegro_hand_shapes + 2

        self.envs = []

        self.object_init_state = []

        self.hand_indices = []
        self.object_indices = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(allegro_hand_rb_count, allegro_hand_rb_count + object_rb_count))

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 0)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(self.randomize_scale_list[i % num_scales] - 0.025, self.randomize_scale_list[i % num_scales] + 0.025)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0
        self.cum_yaw_buf[env_ids] = 0.0         # 清零累计角度
        self.ep_has_succeeded[env_ids] = False  # 清掉成功标记
        self.ep_angvel_sum[env_ids] = 0.0
        self.ep_step_count[env_ids] = 0
        self.spin_time_buf[env_ids] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, 16:]
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
            pos = sampled_pose[:, :16]
            self.allegro_hand_dof_pos[s_ids, :] = pos
            self.allegro_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, :self.num_allegro_hand_dofs] = pos
            self.cur_targets[s_ids, :self.num_allegro_hand_dofs] = pos
            self.init_pose_buf[s_ids, :] = pos.clone()

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

        self.cum_yaw_buf[env_ids] = 0.0         # 清零累计角度

    def compute_observations(self):
        self._refresh_gym()
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.allegro_hand_dof_pos.shape) * 2.0 - 1.0) * self.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.allegro_hand_dof_pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
            self.allegro_hand_dof_pos[at_reset_env_ids], self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.allegro_hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        self.obs_buf[:, :t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())

### LLM
    def compute_reward(self, actions):
        """新的 5 项奖励/惩罚"""
        dt = self.control_freq_inv * self.dt
        self.rot_axis_buf[:, -1] = -1                       # 目标轴 z−

        # ---------- ① 旋转速度奖励 ----------
        angdiff = quat_to_axis_angle(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / dt                        # [B,3]
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(
            vec_dot, min=self.angvel_clip_min, max=self.angvel_clip_max)

        # ---------- ② 角加速度平滑 ----------
        ang_acc = (object_angvel - self.object_angvel_prev) / dt
        angacc_penalty = (ang_acc ** 2).sum(-1)             # L2²

        # ---------- ③ 物体线速度惩罚 ----------
        object_linvel = (self.object_pos - self.object_pos_prev) / dt
        obj_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)

        # ---------- ④ 关节角速度惩罚 ----------
        joint_vel_penalty = (self.allegro_hand_dof_vel ** 2).sum(-1)

        # ---------- ⑤ 整圈计数奖励 ----------
        yaw_delta = vec_dot * dt                            # 绕目标轴累计角度
        self.cum_yaw_buf += yaw_delta
        full_rot = torch.floor(self.cum_yaw_buf / (2 * math.pi))
        full_rot_bonus = full_rot * self.full_rotation_bonus
        self.cum_yaw_buf -= full_rot * 2 * math.pi          # 取余留下残角

        # ---------- 汇总 ----------
        self.rew_buf[:] = (
            self.rotate_reward_scale      * rotate_reward
        - self.angacc_penalty_scale     * angacc_penalty
        - self.object_linvel_penalty_scale * obj_linvel_penalty
        - self.joint_vel_penalty_scale  * joint_vel_penalty
        + full_rot_bonus
        )

        # ---------- 续存 & 终止 ----------
        self.object_angvel_prev[:] = object_angvel          # 更新缓存
        self.reset_buf[:] = self.check_termination(self.object_pos)

        if (self.progress_buf[0] % 100) == 0:
            max_angvel = object_angvel.norm(dim=-1).max()
            tprint(f"Max angular velocity: {max_angvel:.4f} rad/s")

            self.extras['object_angvel_mag'] = object_angvel.norm(dim=-1).mean()
            tprint(f'step {int(self.progress_buf[0])} | ' f'⟨|ω|⟩ = {self.extras["object_angvel_mag"]:.3f} rad/s')

        # ---------- 监控指标 ----------
        self.extras.update({
            'rotate_reward':          rotate_reward.mean(),
            'angacc_penalty':         angacc_penalty.mean(),
            'obj_linvel_penalty':     obj_linvel_penalty.mean(),
            'joint_vel_penalty':      joint_vel_penalty.mean(),
            'full_rot_bonus':         full_rot_bonus.mean(),
        })


#### Simple RL
    def compute_reward(self, actions):
        self.rot_axis_buf[:, 0] = 0  # 设置 x 轴方向
        self.rot_axis_buf[:, 1] = 0  # 设置 y 轴方向
        self.rot_axis_buf[:, 2] = -1
        # pose diff penalty
        pose_diff_penalty = ((self.allegro_hand_dof_pos - self.init_pose_buf) ** 2).sum(-1)
        # work and torque penalty
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2
        # Compute offset in radians. Radians -> radians / sec
        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
        # 打印最大角速度（以便调试）
        if (self.progress_buf[0] % 100) == 0:
            max_angvel = object_angvel.norm(dim=-1).max()
            tprint(f"Max angular velocity: {max_angvel:.4f} rad/s")

        # linear velocity: use position difference instead of self.object_linvel
        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)
        dt_step = self.control_freq_inv * self.dt          # 每步的物理时间
        yaw_delta = vec_dot * dt_step                      # 本步增量（弧度）
        self.cum_yaw_buf += yaw_delta                      # 按 env 累计

        # 若本局尚未记为成功，且累计旋转 ≥ 2π，则记为本局成功

        # newly_succeeded = (~self.ep_has_succeeded) & (self.cum_yaw_buf >= self.SUCCESS_ANGLE_RAD)
        # self.ep_has_succeeded |= newly_succeeded

        if self.require_sign:
            turning_mag = vec_dot                    # 只认固定方向
        else:
            turning_mag = torch.abs(vec_dot)         # 只要在转，不限方向

        turning_cond = turning_mag >= self.success_omega_thresh  # [B] 是否算“在转”

        # 满足条件则累计时间，否则清零
        self.spin_time_buf = torch.where(
            turning_cond, self.spin_time_buf + dt_step, torch.zeros_like(self.spin_time_buf)
        )

        # 只有“本局尚未成功”的 env 才会在达标时置成功
        newly_succeeded = (~self.ep_has_succeeded) & (self.spin_time_buf >= self.success_spin_time)
        self.ep_has_succeeded |= newly_succeeded

        self.rew_buf[:] = compute_hand_reward(
            object_linvel_penalty, self.object_linvel_penalty_scale,
            rotate_reward, self.rotate_reward_scale,
            pose_diff_penalty, self.pose_diff_penalty_scale,
            torque_penalty, self.torque_penalty_scale,
            work_penalty, self.work_penalty_scale,
        )
        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['object_linvel_penalty'] = object_linvel_penalty.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = object_angvel[:, 0].mean()
        self.extras['pitch'] = object_angvel[:, 1].mean()
        self.extras['yaw'] = object_angvel[:, 2].mean()

        self.extras['object_angvel_mag'] = object_angvel.norm(dim=-1).mean()
        if (self.progress_buf[0] % 100) == 0:          # progress_buf 第 0 个 env 就够
            tprint(f'step {int(self.progress_buf[0])} | ' f'⟨|ω|⟩ = {self.extras["object_angvel_mag"]:.3f} rad/s')

        if self.evaluate:
            finished_episode_mask = self.reset_buf == 1
            self.stat_sum_rewards += self.rew_buf.sum()
            self.stat_sum_rotate_rewards += rotate_reward.sum()
            self.stat_sum_torques += self.torques.abs().sum()
            self.stat_sum_obj_linvel += (self.object_linvel ** 2).sum(-1).sum()
            self.stat_sum_episode_length += (self.reset_buf == 0).sum()
            self.env_evaluated += (self.reset_buf == 1).sum()
            self.env_timeout_counter[finished_episode_mask] += 1
            info = f'progress {self.env_evaluated} / {self.max_evaluate_envs} | ' \
                   f'reward: {self.stat_sum_rewards / self.env_evaluated:.2f} | ' \
                   f'eps length: {self.stat_sum_episode_length / self.env_evaluated:.2f} | ' \
                   f'rotate reward: {self.stat_sum_rotate_rewards / self.env_evaluated:.2f} | ' \
                   f'lin vel (x100): {self.stat_sum_obj_linvel * 100 / self.stat_sum_episode_length:.4f} | ' \
                   f'command torque: {self.stat_sum_torques / self.stat_sum_episode_length:.2f}'
            tprint(info)
            if self.env_evaluated >= self.max_evaluate_envs:
                exit()

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)
        step_i = int(self.progress_buf[0].item())
        if (step_i % 50) == 0:
            # ω 在 -z 方向的分量
            w_target_all = -self.object_angvel[:, 2]       # shape [num_envs]
            self.ep_w_target_sum += w_target_all
            self.ep_w_target_count += 1
            avg_w_target = w_target_all.mean().item()     # 平均值
            max_w_target = w_target_all.max().item()      # 最大值
            min_w_target = w_target_all.min().item()      # 最小值

            log_str = (f"[step {step_i}] avg ω·(-ẑ) = {avg_w_target:.3f} rad/s "
                    f"(min={min_w_target:.3f}, max={max_w_target:.3f})")

            # 打印到终端
            tprint(log_str)

            # 追加到 txt 文件
            log_file = "outputs/avg_angvel_log.txt"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_str + "\n")
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        
        if len(env_ids) > 0:
            # 1) 计算这些 env 本局的“平均角速度”（每局：sum(|ω·axis|)/steps）
            # 避免除零
            steps = self.ep_step_count[env_ids].clamp(min=1).float()
            per_ep_mean_angvel = (self.ep_angvel_sum[env_ids] / steps)  # [num_finished]
            sum_means = float(per_ep_mean_angvel.sum().item())

            # 2) 成功个数
            succ = self.ep_has_succeeded[env_ids]
            num_finished = int(env_ids.numel())
            num_success  = int(succ.sum().item())

            # 3) 累加到全局
            self.total_episodes += num_finished
            self.total_successes += num_success
            self.sum_episode_mean_angvel += sum_means

            avg_success = (self.total_successes / self.total_episodes) if self.total_episodes > 0 else 0.0
            avg_angvel  = (self.sum_episode_mean_angvel / self.total_episodes) if self.total_episodes > 0 else 0.0

            # 4) 单行稳定可 grep 的输出（适合论文复现实验）
            tprint(f"RESULT avg_angvel_rad_s={avg_angvel:.6f} avg_success={avg_success:.6f} N={self.total_episodes}")

            # 5) 追加到 CSV：episodes, avg_angvel_rad_s, avg_success
            try:
                with open(self.metrics_fpath, "a") as f:
                    f.write(f"{self.total_episodes},{avg_angvel:.6f},{avg_success:.6f}\n")
            except Exception as e:
                tprint(f"[metrics] write csv failed: {e}")

            avg_w_target_ep = (self.ep_w_target_sum[env_ids] / self.ep_w_target_count[env_ids].clamp(min=1)).cpu().numpy()
            self.ep_w_target_sum[env_ids] = 0.0
            self.ep_w_target_count[env_ids] = 0

            # 6) 现在才 reset（会清空本局缓存）
            self.reset_idx(env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        # if self.total_episodes >= 10:
        #     tprint("Single test episode finished. Exiting...")
        # exit()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.prev_targets + 1 / 24 * self.actions
        self.cur_targets[:] = tensor_clamp(targets, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.prev_targets[:] = self.cur_targets.clone()
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.object_angvel_prev[:] = self.object_angvel

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        return self.obs_dict

    def step(self, actions):
        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def update_low_level_control(self):
        previous_dof_pos = self.allegro_hand_dof_pos.clone()
        self._refresh_gym()
        if self.torque_control:
            dof_pos = self.allegro_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torch.clip(torques, -0.5, 0.5).clone()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def check_termination(self, object_pos):
        resets = torch.logical_or(
            torch.less(object_pos[:, -1], self.reset_z_threshold),
            torch.greater_equal(self.progress_buf, self.max_episode_length),
        )
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_scale = rand_config['randomizeScale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.joint_noise_scale = rand_config['jointNoiseScale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.enable_priv_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'mycube': 'assets/mycube50.urdf',
        }
        for p_id, prim in enumerate(primitive_list):
            if 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'../assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))] + ['mycube']
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
                # print('cuboids', cuboids)
                # print('subset_name', subset_name)
                # print('cuboid_list', cuboid_list)
                # print('object_type_prob', self.object_type_prob)
                # print('cuboids', cuboids)
                # print('asset_files_dict', self.asset_files_dict)

            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(self.object_type_list)
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _allocate_task_buffer(self, num_envs):
        self.priv_info_buf = torch.zeros((num_envs, self.num_env_factors), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)

    def _setup_reward_config(self, r_config):
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        self.rotate_reward_scale = r_config['rotateRewardScale']
        self.object_linvel_penalty_scale = r_config['objLinvelPenaltyScale']
        self.pose_diff_penalty_scale = r_config['poseDiffPenaltyScale']
        self.torque_penalty_scale = r_config['torquePenaltyScale']
        self.work_penalty_scale = r_config['workPenaltyScale']

        self.angacc_penalty_scale     = r_config['angAccPenaltyScale']
        self.joint_vel_penalty_scale  = r_config['jointVelPenaltyScale']
        self.full_rotation_bonus      = r_config['fullRotationBonus']

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = True
        hand_asset_options.disable_gravity = False
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        else:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)

        # load object asset
        self.object_asset_list = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)

    def _init_object_pose(self):
        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(0, 0, 0.5)
        allegro_hand_start_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), -np.pi / 2) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x
        pose_dx, pose_dy, pose_dz = -0.01, -0.04, 0.15

        object_start_pose.p.x = allegro_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz

        object_start_pose.p.y = allegro_hand_start_pose.p.y - 0.01
        # for grasp pose generation, it is used to initialize the object
        # it should be slightly higher than the fingertip
        # so it is set to be 0.66 for internal allegro and 0.64 for the public allegro
        # ----
        # for in-hand object rotation, the initialization of z is only used in the first step
        # it is set to be 0.65 for backward compatibility
        object_z = 0.66 if self.save_init_pose else 0.65
        if 'internal' not in self.grasp_cache_name:
            object_z -= 0.02
        object_start_pose.p.z = object_z
        return allegro_hand_start_pose, object_start_pose


def compute_hand_reward(
    object_linvel_penalty, object_linvel_penalty_scale: float,
    rotate_reward, rotate_reward_scale: float,
    pose_diff_penalty, pose_diff_penalty_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
):
    reward = rotate_reward_scale * rotate_reward
    # Distance from the hand to the object
    reward = reward + object_linvel_penalty * object_linvel_penalty_scale
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    return reward


def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 3:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :3] / sin_half_angles_over_angles
