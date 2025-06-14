"""Defines simple task for training a joystick walking policy for K-Bot."""

import asyncio
import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import numpy as np
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from mujoco_animator import MjAnim

# These are in the order of the neural network outputs.
# Joint name, target position, penalty weight.
ZEROS: list[tuple[str, float, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0, 1.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0), 1.0),
    ("dof_right_shoulder_yaw_02", 0.0, 1.0),
    ("dof_right_elbow_02", math.radians(90.0), 1.0),
    ("dof_right_wrist_00", 0.0, 1.0),
    ("dof_left_shoulder_pitch_03", 0.0, 1.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0), 1.0),
    ("dof_left_shoulder_yaw_02", 0.0, 1.0),
    ("dof_left_elbow_02", math.radians(-90.0), 1.0),
    ("dof_left_wrist_00", 0.0, 1.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0), 1.0),
    ("dof_right_hip_roll_03", math.radians(-0.0), 2.0),
    ("dof_right_hip_yaw_03", 0.0, 2.0),
    ("dof_right_knee_04", math.radians(-50.0), 1.0),
    ("dof_right_ankle_02", math.radians(30.0), 1.0),
    ("dof_left_hip_pitch_04", math.radians(20.0), 1.0),
    ("dof_left_hip_roll_03", math.radians(0.0), 2.0),
    ("dof_left_hip_yaw_03", 0.0, 2.0),
    ("dof_left_knee_04", math.radians(50.0), 1.0),
    ("dof_left_ankle_02", math.radians(-30.0), 1.0),
]


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Task parameters.
    reference_motion_path: Path = xax.field(
        value="dance_kawaii.mjanim",
        help="The path to the reference motion to use for the task.",
    )

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the RNN.",
    )
    depth: int = xax.field(
        value=2,
        help="The depth for the RNN.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    use_gyro: bool = xax.field(
        value=True,
        help="Whether to use the IMU gyroscope observations.",
    )
    gait_freq_range: tuple[float, float] = xax.field(
        value=(1.2, 1.5),
        help="The range of gait frequencies to use.",
    )

    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=30,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=30.0,
        help="Increase the curriculum level when the mean trajectory length is above this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=10.0,
        help="Decrease the curriculum level when the mean trajectory length is below this threshold.",
    )
    min_level_steps: int = xax.field(
        value=10,
        help="The minimum number of steps to wait before changing the curriculum level.",
    )
    min_level: float = xax.field(
        value=0.01,
        help="The minimum curriculum level.",
    )

    # Reward Weights
    action_acc: float = xax.field(
        value=0.02,
        help="The weight for the action acceleration penalty.",
    )
    action_vel: float = xax.field(
        value=0.02,
        help="The weight for the action velocity penalty.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


@attrs.define(frozen=True, kw_only=True)
class FrameTimestepObservation(ksim.TimestepObservation):
    """Observation of the timestep mod the length of the motion reference."""

    motion_reference: ksim.MotionReferenceData

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        timestep = super().observe(state, curriculum_level, rng)

        return jnp.mod(timestep, self.motion_reference.num_frames * self.motion_reference.ctrl_dt)


def quat_dot(q1, q2):
    """Dot product shape (...,4)."""
    return jnp.sum(q1 * q2, axis=-1)


def quat_conj(q):
    """Conjugate keeps w, flips xyz."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a, b):
    """Hamilton product shape (...,4)."""
    w1, x1, y1, z1 = jnp.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = jnp.moveaxis(b, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.stack([w, x, y, z], axis=-1)


def quat_angle_error(q, q_ref):
    """
    Smallest angular distance (rad) between two unit quats.
    """
    # Use dot, deal with double cover by abs, clamp for numerical safety
    cos_half_theta = jnp.clip(jnp.abs(quat_dot(q, q_ref)), 0.0, 1.0)
    return 2.0 * jnp.arccos(cos_half_theta)


def angle_to_reward(angle, sharpness=5.0):
    return jnp.exp(-sharpness * angle**2)


@attrs.define(frozen=True, kw_only=True)
class QposReferenceMotionReward(ksim.Reward):
    """Reward for matching the reference motion."""

    scale: float = 1.0
    reference_motion: ksim.MotionReferenceData
    # BUG: Quaternion similarity is not just per component

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        full_qpos_ref = self.reference_motion.get_qpos_at_time(trajectory.timestep)
        qpos_ref_joints = full_qpos_ref[:, 7:]
        qpos_joints = trajectory.qpos[:, 7:]

        diff = qpos_joints - qpos_ref_joints
        joint_pos_reward = ksim.norm_to_reward(xax.get_norm(diff, "l2")).mean(axis=-1)

        # Quaternion similarity
        quat_ref = full_qpos_ref[:, 3:7]
        quat = trajectory.qpos[:, 3:7]

        ang_err = quat_angle_error(quat, quat_ref)  # shape (batch,)
        quat_r = angle_to_reward(ang_err, sharpness=5)

        # Root body reward
        root_pos_ref = full_qpos_ref[
            :, :3
        ]  # verified that the trajectory I am testing with starts at 0,0 so this should be fine
        root_pos = trajectory.qpos[:, :3]
        root_pos_diff = root_pos - root_pos_ref
        root_pos_reward = ksim.norm_to_reward(xax.get_norm(root_pos_diff, "l2")).mean(axis=-1)

        total_reward = joint_pos_reward + quat_r + root_pos_reward
        return total_reward


@attrs.define(frozen=True, kw_only=True)
class QvelReferenceMotionReward(ksim.Reward):
    """Reward for matching the reference motion."""

    scale: float = 1.0
    reference_motion: ksim.MotionReferenceData

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        step = jnp.round(trajectory.timestep / self.reference_motion.ctrl_dt).astype(int)
        max_qvel_step = self.reference_motion.qvel.array.shape[0] - 1
        safe_step = jnp.clip(step, 0, max_qvel_step)

        qvel_ref = self.reference_motion.get_qvel_at_step(safe_step)[:, 6:]
        qvel = trajectory.qvel[:, 6:]

        # Debug: Check for NaN/inf in inputs
        qvel_safe = jnp.nan_to_num(qvel, nan=0.0, posinf=1e6, neginf=-1e6)
        qvel_ref_safe = jnp.nan_to_num(qvel_ref, nan=0.0, posinf=1e6, neginf=-1e6)

        qvel_diff = qvel_safe - qvel_ref_safe

        # Clip extreme values to prevent overflow in norm computation
        qvel_diff = jnp.clip(qvel_diff, -1e3, 1e3)

        norm_val = xax.get_norm(qvel_diff, "l2")
        # Ensure norm is finite
        norm_val = jnp.clip(norm_val, 1e-8, 1e3)

        joint_vel_reward = ksim.norm_to_reward(norm_val).mean(axis=-1)

        # calculate difference in root's velocity, linear and angular. Think we can just use l2 here
        root_vel_ref = self.reference_motion.get_qvel_at_step(safe_step)[:, :6]
        root_vel = trajectory.qvel[:, :6]
        root_vel_diff = root_vel - root_vel_ref
        root_vel_reward = ksim.norm_to_reward(xax.get_norm(root_vel_diff, "l2")).mean(axis=-1)

        total_reward = joint_vel_reward + root_vel_reward
        return total_reward


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(self.num_outputs, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(self.num_outputs, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(self.num_outputs, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v, _ in ZEROS])[:, None]

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
        model = mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")
        names_to_idxs = ksim.get_geom_data_idx_by_name(model)
        model.geom_priority[names_to_idxs["floor"]] = 2.0
        return model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot-headless"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
            action_noise=math.radians(5),
            action_noise_type="gaussian",
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            # ksim.StaticFrictionRandomizer(),
            # ksim.ArmatureRandomizer(scale_lower=0.1, scale_upper=10.0),
            # ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.85, scale_upper=1.15),
            # ksim.JointDampingRandomizer(scale_lower=0.1, scale_upper=10.0),
            # ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-4), scale_upper=math.radians(4)),
            # ksim.FloorFrictionRandomizer.from_geom_name(
            #     model=physics_model, floor_geom_name="floor", scale_lower=0.3, scale_upper=1.5
            # ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v, _ in ZEROS}, scale=math.radians(45)),
            ksim.RandomJointVelocityReset(),
            ksim.RandomHeightReset(range=(0.0, 0.3)),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            # Corresponds to "clock" in mimic tasks
            FrameTimestepObservation(motion_reference=self.reference_motion),
            # Corresponds to qpos
            ksim.JointPositionObservation(noise=math.radians(3)),
            # Corresponds to qvel
            ksim.JointVelocityObservation(noise=math.radians(90)),
            # Corresponds to root z-pos
            ksim.BasePositionObservation(),
            # Corresponds to root quat
            ksim.BaseOrientationObservation(),
            # Corresponds to root lin vel
            ksim.BaseLinearVelocityObservation(),
            # Corresponds to root ang vel
            ksim.BaseAngularVelocityObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return []

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            QposReferenceMotionReward(
                scale=1.0,
                reference_motion=self.reference_motion,
            ),
            QvelReferenceMotionReward(scale=0.2, reference_motion=self.reference_motion),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.3, unhealthy_z_upper=10.0),
            ksim.NotUprightTermination(max_radians=math.radians(60)),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.EpisodeLengthCurriculum(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
            min_level=self.config.min_level,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        num_joints = len(ZEROS)

        # Calculate observation size
        # root_z (1) + root_quat (4) + joint_pos (N) + root_lin_vel (3) + root_ang_vel (3) + joint_vel (N)
        # + ref_qpos (N+7) + ref_qvel (N+6)
        num_obs = 1 + 4 + num_joints + 3 + 3 + num_joints
        num_ref_obs = (num_joints + 7) + (num_joints + 6)
        num_inputs = num_obs + num_ref_obs

        return Model(
            key,
            num_actor_inputs=num_inputs,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=num_inputs,  # Critic is not privileged
            min_std=0.01,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def _get_obs_vec(
        self,
        observations: xax.FrozenDict[str, Array],
    ) -> Array:
        # ---- Base Observations ----
        # from KBotV2FlatFoot._get_observation_specification
        # ObservationType.FreeJointPosNoXY -> root_z, root_quat
        # ObservationType.JointPos -> joint_pos_n
        # ObservationType.FreeJointVel -> base_lin_vel_3, base_ang_vel_3
        # ObservationType.JointVel -> joint_vel_n
        root_z_1 = observations["base_position_observation"][..., 2:3]
        root_quat_4 = observations["base_orientation_observation"]
        joint_pos_n = observations["joint_position_observation"]
        base_lin_vel_3 = observations["base_linear_velocity_observation"]
        base_ang_vel_3 = observations["base_angular_velocity_observation"]
        joint_vel_n = observations["joint_velocity_observation"]

        # ---- Reference Motion Observations ----
        timestep_1 = observations["frame_timestep_observation"]
        ref_qpos = self.reference_motion.get_qpos_at_time(timestep_1).squeeze(0)
        step = jnp.round(timestep_1 / self.reference_motion.ctrl_dt).astype(int)
        max_qvel_step = self.reference_motion.qvel.array.shape[0] - 1
        safe_step = jnp.clip(step, 0, max_qvel_step)
        ref_qvel = self.reference_motion.get_qvel_at_step(safe_step).squeeze(0)

        obs = [
            root_z_1,
            root_quat_4,
            joint_pos_n,
            base_lin_vel_3,
            base_ang_vel_3,
            joint_vel_n,
            ref_qpos,
            ref_qvel,
        ]

        return jnp.concatenate(obs, axis=-1)

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        obs_n = self._get_obs_vec(observations)
        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        obs_n = self._get_obs_vec(observations)
        return model.forward(obs_n, carry)

    def _ppo_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(rng),
            (next_actor_carry, next_critic_carry),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._ppo_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))

    def run(self) -> None:
        animation = MjAnim.load(self.config.reference_motion_path)
        qpos_sequence = animation.to_numpy(self.config.ctrl_dt, interp="cubic", loop=True)

        z_offset = -0.03
        qpos_sequence[:, 2] += z_offset

        mj_model = self.get_mujoco_model()
        qvel_list = []
        for i in range(len(qpos_sequence) - 1):
            qvel = np.zeros(mj_model.nv)
            mujoco.mj_differentiatePos(mj_model, qvel, self.config.ctrl_dt, qpos_sequence[i], qpos_sequence[i + 1])
            qvel_list.append(qvel)
        qvel_sequence = jnp.array(qvel_list)

        self.reference_motion = ksim.MotionReferenceData(
            qpos=xax.HashableArray(qpos_sequence[:-1]),
            qvel=xax.HashableArray(qvel_sequence),
            cartesian_poses=xax.FrozenDict({}),
            ctrl_dt=self.config.ctrl_dt,
        )

        if self.config.run_mode.lower() == "view_motion":
            print(self.reference_motion.qpos.array[0])
            ksim.visualize_reference_motion(
                model=self.get_mujoco_model(),
                reference_qpos=np.asarray(self.reference_motion.qpos.array),
                cartesian_motion=xax.FrozenDict(
                    {
                        body_id: np.asarray(poses.array)
                        for body_id, poses in self.reference_motion.cartesian_poses.items()
                    }
                ),
                mj_base_id=0,
                ctrl_dt=0.02,
            )
        else:
            super().run()


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2,
            batch_size=1,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            global_grad_clip=2.0,
            learning_rate=1e-3,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.001, 0.01),  # Simulate 3-5ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_track_body_id=0,
            # Checkpointing parameters.
            save_every_n_seconds=60,
        ),
    )
