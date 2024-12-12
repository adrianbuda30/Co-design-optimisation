import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import os  # For handling file paths
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class QuadcopterEnv_Euler(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        env_id=0,
        forward_reward_weight=1.0,
        xml_file=f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/quadcopter.xml",
        ctrl_cost_weight=1e-3,
        render_mode='human',
        healthy_reward=1.0,
        healthy_z_range=(-4.0, 10.0),
        healthy_angle_range=(-0.5, 0.5),
        reset_noise_scale=0.0,
        exclude_current_positions_from_observation=False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            render_mode,
            healthy_reward,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self.env_id = env_id
        self.design_params = np.ones(8)
        xml_file_update = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/quadcopter_{self.env_id}.xml"
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight


        self._healthy_reward = healthy_reward


        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range


        self._reset_noise_scale = reset_noise_scale

        # Physical parameters
        self.mass = 1.0  # kg
        self.g = 9.81  # m/s^2
        self.I = np.array([0.1, 0.1, 0.15])  # moment of inertia
        self.arm_length = 0.2  # m
        self.max_thrust = 2 * self.mass * self.g  # N
        self.max_torque = 1.0  # N⋅m

        # Environment parameters
        self.target = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        obs_shape = 20
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_shape + 1,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            xml_file_update,
            4,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )



    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()


        design_params = self.design_params
        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity, design_params)).ravel()
        return observation

    def _apply_dynamics(self, thrust, torques):
        # Extract state variables
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        angles = self.data.qpos[4:7]
        angular_rates = self.data.qvel[3:6]

        # Rotation matrix from body to inertial frame
        cr, cp, cy = np.cos(angles)
        sr, sp, sy = np.sin(angles)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

        # Forces (in inertial frame)
        thrust_force = R @ np.array([0, 0, thrust * self.max_thrust])
        gravity_force = np.array([0, 0, -self.mass * self.g])
        total_force = thrust_force + gravity_force

        # Accelerations
        linear_acc = total_force / self.mass

        # Angular acceleration
        torques = torques * self.max_torque
        angular_acc = torques / self.I

        # Update state with semi-implicit Euler integration
        self.data.qvel[:3] += linear_acc * self.dt
        self.data.qvel[3:6] += angular_acc * self.dt

        self.data.qpos[:3] += self.data.qvel[:3] * self.dt
        self.data.qpos[4:7] += self.data.qvel[3:6] * self.dt

        # Normalize angles to [-pi, pi]
        self.data.qpos[4:7] = np.mod(self.data.qpos[4:7] + np.pi, 2 * np.pi) - np.pi

    def rotor_torques_to_actuations(self, rotor_torques, arm_length, thrust_coefficient, yaw_coefficient):
        """
        Converts rotor torques to high-level controls: thrust and torques.

        Parameters:
        - rotor_torques: [tau1, tau2, tau3, tau4]
        - arm_length: Distance from the quadcopter's center to each rotor.
        - thrust_coefficient: Coefficient to convert torques to collective thrust.
        - yaw_coefficient: Coefficient to convert torques to yaw torque.

        Returns:
        - High-level controls [thrust, roll_torque, pitch_torque, yaw_torque]
        """
        tau1, tau2, tau3, tau4 = rotor_torques

        # Compute high-level controls
        thrust = thrust_coefficient * (tau1 + tau2 + tau3 + tau4)
        roll_torque = arm_length * (-tau1 + tau2 + tau3 - tau4)
        pitch_torque = arm_length * (-tau1 - tau2 + tau3 + tau4)
        yaw_torque = yaw_coefficient * (-tau1 + tau2 - tau3 + tau4)

        return np.array([thrust, roll_torque, pitch_torque, yaw_torque])

    def step(self, action):
        #self.render_mode = "human"

        prev_distance = np.linalg.norm(self.data.qpos[:3] - self.target)

        arm_length = self.design_params[0]
        thrust_coefficient = 1.0
        yaw_coefficient = 0.1

        high_level_controls = self.rotor_torques_to_actuations(action, arm_length, thrust_coefficient, yaw_coefficient)

        # Apply dynamics
        thrust = high_level_controls[0]
        torques = high_level_controls[1:]
        self._apply_dynamics(thrust, torques)

        observation = self._get_obs()

        current_distance = np.linalg.norm(self.data.qpos[:3] - self.target)

        # Calculate reward using new method
        reward = self._calculate_reward(current_distance, prev_distance, high_level_controls)

        # Check termination conditions
        done = False

        z = self.data.qpos[2]
        angle_x = self.data.qpos[4]
        angle_y = self.data.qpos[5]
        angle_z = self.data.qpos[6]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle_x < max_angle and min_angle < angle_y < max_angle and min_angle < angle_z < max_angle
        is_healthy = healthy_z and healthy_angle

        if current_distance < 0.05:
            reward += 200  # Final success bonus
            done = False

        if not is_healthy:
            reward -= 200  # Penalty for failure
            done = True

        info = {
            "position": self.data.qpos[:3],
            "velocity": self.data.qvel[:3]
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, info


    def _calculate_reward(self, current_distance, prev_distance, action):
        reward = 0

        # Distance improvement reward
        distance_improvement = prev_distance - current_distance
        reward += distance_improvement * 20

        # Smooth progress reward
        progress = (self.initial_distance - current_distance) / self.initial_distance
        reward += np.exp(-current_distance) * 5  # Exponential reward for getting closer

        # Hovering reward when near target
        if current_distance < 0.3:
            hover_stability = 1.0 / (1.0 + np.sum(np.abs(self.data.qvel[:3])))  # Velocity stability
            reward += hover_stability * 10

        # Orientation penalties
        angle_penalty = -0.1 * np.sum(np.abs(self.data.qpos[4:7]))
        angular_rate_penalty = -0.05 * np.sum(np.abs(self.data.qvel[3:6]))
        reward += angle_penalty + angular_rate_penalty

        # Energy efficiency reward/penalty
        energy_penalty = -0.1 * (action[0] + 0.5 * np.sum(np.abs(action[1:])))
        reward += energy_penalty

        # Small constant penalty to encourage faster completion
        reward -= 0.1

        return reward


    def reset_model(self, seed=1):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        self.initial_distance = np.linalg.norm(self.data.qpos[:3] - self.target)

        observation = self._get_obs()
        return observation

    def set_design_params(self, design_params):
        self.design_params = design_params

    def set_env_id(self, env_id):
        self.env_id = env_id

    def get_design_params(self):
        return self.design_params

    def get_env_id(self):
        return self.env_id

