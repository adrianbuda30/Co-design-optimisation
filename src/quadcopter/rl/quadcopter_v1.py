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


class QuadcopterEnv(MujocoEnv, utils.EzPickle):
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
        self.target = np.array([5.0, 0.0, 5.0], dtype=np.float32)

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


    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost


    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()


        design_params = self.design_params
        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity, design_params)).ravel()
        return observation

    def step(self, action):
        self.render_mode = "human"

        prev_distance = np.linalg.norm(self.data.qpos[:3] - self.target)

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        current_distance = np.linalg.norm(self.data.qpos[:3] - self.target)

        # Calculate reward using new method
        reward = self._calculate_reward(current_distance, prev_distance, action)

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

        # Add condition for the spherical area
        forbidden_center = np.array([2.5, 0, 2.5])
        forbidden_radius = 1.5
        distance_to_forbidden = np.linalg.norm(self.data.qpos[:3] - forbidden_center)
        in_forbidden_area = distance_to_forbidden < forbidden_radius

        if current_distance < 0.05:
            reward += 200  # Final success bonus
            done = False

        if not is_healthy or in_forbidden_area:
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
        energy_penalty = -0.1 * (np.sum(np.abs(action[0:])))
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



