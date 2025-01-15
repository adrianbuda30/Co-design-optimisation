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
        "render_fps": 125,
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
        healthy_angle_range=(-1.5, 1.5),
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

        self.current_gate_index = 0


        self._healthy_reward = healthy_reward

        self.gates = [
            np.array([-4, 4, 0]),
            np.array([0, 8, 0]),
            np.array([4, 4, 0]),
            np.array([0, 0, 0])
            #np.array([-4.5, -6, 1]),
            #np.array([4.5, -0.5, 1]),
            #np.array([-2, 7, 1]),
            #np.array([-1, 1, 3.5])
        ]


        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range


        self._reset_noise_scale = reset_noise_scale

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
        #self.render_mode = "human"

        if not hasattr(self, 'prev_position'):
            self.prev_position = self.data.qpos[:3].copy()

        # Get the current gate and position
        current_gate = self.gates[self.current_gate_index]
        current_position = self.data.qpos[:3]


        # Body rate (angular velocity)
        body_rate = self.data.qvel[3:6]  # Assuming these are angular velocities


        # Calculate reward
        reward = self._calculate_reward(
            current_position=current_position,
            prev_position=self.prev_position,
            current_gate=current_gate,
            body_rate=body_rate,
            action=action
        )

        done = False


        # Update previous position
        self.prev_position = current_position.copy()

        # Simulate the quadcopter's action
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        # Update gate if passed
        gate_passed_threshold = 0.5  # Threshold to consider a gate passed
        if np.linalg.norm(current_gate - current_position) < gate_passed_threshold:
            self.current_gate_index += 1

            if self.current_gate_index == len(self.gates) - 1:  # All gates passed
                done = True
                return self._get_obs(), reward, done, False, {}


        z = self.data.qpos[2]
        angle_x = self.data.qpos[4]
        angle_y = self.data.qpos[5]
        angle_z = self.data.qpos[6]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = (
                min_angle < angle_x < max_angle and
                min_angle < angle_y < max_angle and
                min_angle < angle_z < max_angle
        )
        if not (healthy_z and healthy_angle):
            reward -= 50.0
            done = True

        info = {
            "position": current_position,
            "velocity": self.data.qvel[:3],
            "current_gate": self.current_gate_index
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, info

    def _calculate_reward(self, current_position, prev_position, current_gate, body_rate, action):
        """
        Calculate the reward based on progress through gates, body rate

        Args:
            current_position (np.array): Current position of the quadcopter.
            prev_position (np.array): Position of the quadcopter at the previous timestep.
            current_gate (np.array): Coordinates of the target gate.
            body_rate (np.array): Angular velocity of the quadcopter.
            action (np.array): Current action applied to the quadcopter.

        Returns:
            float: Calculated reward.
        """
        # Parameters
        body_rate_penalty_coeff = 0.01
        finish_reward = 50.0
        gate_reward = 10.0

        # Distance-based progress reward
        prev_distance_to_gate = np.linalg.norm(current_gate - prev_position)
        current_distance_to_gate = np.linalg.norm(current_gate - current_position)
        progress_reward = prev_distance_to_gate - current_distance_to_gate

        # Penalty for body rates (smoothness)
        body_rate_penalty = body_rate_penalty_coeff * np.linalg.norm(body_rate)


        # Total reward
        reward = progress_reward - body_rate_penalty

        if current_distance_to_gate < 0.5 and self.current_gate_index < len(self.gates) - 1:
            reward += gate_reward

        # Add final reward if the last gate is passed
        if current_distance_to_gate < 0.5 and self.current_gate_index == len(self.gates) - 1:
            reward += finish_reward



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

        self.current_gate_index = 0

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



