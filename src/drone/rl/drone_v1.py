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


class DroneEnv(MujocoEnv, utils.EzPickle):
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
        xml_file=f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/assets/drone.xml",
        ctrl_cost_weight=1e-3,
        render_mode='human',
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(-200.0, 1000.0),
        healthy_angle_range=(-45.0, 45.0),
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
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self.env_id = env_id
        self.design_params = np.ones(3)
        xml_file_update = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/assets/drone_{self.env_id}.xml"
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight


        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range


        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        obs_shape = 18
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


    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z = self.data.qpos[0]
        angle = self.data.qpos[3]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

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
        y_position_before = self.data.qpos[1]

        site_id = self.get_site_id(self.model, 'left_wing')
        body_id = 2 #self.model.site_bodyid[site_id]
        force = np.array([2000.0, 0.0, 0.5 * 1.225 * 6.28 * np.abs(self.data.qpos[6]) * 2.0 * 0.5 * self.data.qvel[2] ** 2])
        self.data.xfrc_applied[body_id, :3] = force

        site_id = self.get_site_id(self.model, 'right_wing')
        body_id = 3 #self.model.site_bodyid[site_id]
        force = np.array([2000.0, 0.0, 0.5 * 1.225 * 6.28 * np.abs(self.data.qpos[7]) * 2.0 * 0.5 * self.data.qvel[2] ** 2])
        self.data.xfrc_applied[body_id, :3] = force

        self.do_simulation(action, self.frame_skip)


        y_position_after = self.data.qpos[1]

        z_position = 1 / (1 + self.data.qpos[0])
        y_velocity = (y_position_after - y_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = z_position #self._forward_reward_weight * y_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()

        reward = rewards - costs
        terminated = self.terminated
        info = {
            "x_position": y_position_after,
            "x_velocity": y_velocity,
        }


        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info


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

    def get_site_id(self, model, site_name):
        # Split the names into a list
        names = model.names.decode('utf-8').split('\x00')
        # Find the index of the site by name
        for i, name in enumerate(names):
            if name == site_name:
                return i
        raise ValueError(f"Site with name '{site_name}' not found.")


