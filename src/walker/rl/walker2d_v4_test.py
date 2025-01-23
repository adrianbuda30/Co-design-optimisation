import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class Walker2dEnv(MujocoEnv, utils.EzPickle):
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
        xml_file=f"/home/ab2419/Co-design-optimisation/src/walker/assets/walker2d.xml",
        ctrl_cost_weight=1e-3,
        render_mode='human',
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 5.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=0.0,
        track_length=10,
        exclude_current_positions_from_observation=True,
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
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight


        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self.track_length = track_length
        self.start_position = 0

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range


        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        obs_shape = 17
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
            xml_file,
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
        z, angle = self.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy


    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        #self.render_mode = "human"
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()

        reward = rewards - costs

        if x_position_after >= self.start_position + self.track_length:
            terminated = True
            reward += 1000
        elif self._terminate_when_unhealthy and not self.is_healthy:
            terminated = True
            reward -= 100
        else:
            terminated = False

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
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

        self.start_position = self.data.qpos[0]

        observation = self._get_obs()
        return observation

    def set_env_id(self, env_id):
        self.env_id = env_id

    def get_env_id(self):
        return self.env_id

