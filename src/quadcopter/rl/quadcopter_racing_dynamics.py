import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class QuadcopterEnv(gym.Env):
    def __init__(self,
                 env_id = 0):
        super(QuadcopterEnv, self).__init__()

        # Actions: [thrust, roll_torque, pitch_torque, yaw_torque]
        self.action_space = Box(
            low=np.array([0, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )

        self.design_params = np.ones(8)
        self.current_gate_index = 0
        self.num_future_gates = 2

        self.healthy_z_range = (-10.0, 10.0)
        self.healthy_angle_range = (-np.pi / 2, np.pi / 2)



        self.gates = [
            np.array([-4, 4, 0]),
            np.array([0, 8, 0]),
            np.array([4, 12, 0]),
            np.array([0, 16, 0])
            #np.array([-4.5, -6, 1]),
            #np.array([4.5, -0.5, 1]),
            #np.array([-2, 7, 1]),
            #np.array([-1, 1, 3.5])
        ]


        obs_shape = 20 + self.num_future_gates * 4 * 3

        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
            )

        self.mass = 1.0  # kg
        self.g = 9.81  # m/s^2
        self.I = np.array([0.01, 0.01, 0.015])  # moment of inertia
        self.arm_length = 0.05  # m

        self.target = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        self.dt = 0.1  # smaller timestep for better dynamics
        self.max_episode_steps = 500
        self.current_step = 0

        self.state = np.zeros(12, dtype=np.float32)

        self.env_id = env_id
    def _get_obs(self):
        position = self.state[:3]

        design_params = self.design_params

        gate_obs = []
        for i in range(self.num_future_gates):
            gate_index = self.current_gate_index + i
            if gate_index < len(self.gates):
                # Get the corners of the current gate
                gate_corners = self._get_gate_corners(self.gates[gate_index])

                # Calculate the relative position to each corner
                for corner in gate_corners:
                    delta_p = corner - position[:3]
                    gate_obs.append(delta_p)

            else:
                # If there are no more gates, fill with zeros
                gate_obs.extend([np.zeros(3)] * 4)

        gate_obs = np.array(gate_obs).flatten()

        observation = np.concatenate((self.state, design_params, gate_obs)).ravel()
        return observation

    def _apply_dynamics(self, thrust, torques):
        # Extract state variables
        angles = self.state[6:9]

        # Rotation matrix from body to inertial frame
        cr, cp, cy = np.cos(angles)
        sr, sp, sy = np.sin(angles)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

        # Forces (in inertial frame)
        thrust_force = R @ np.array([0, 0, thrust])
        gravity_force = np.array([0, 0, -self.mass * self.g])
        total_force = thrust_force + gravity_force

        # Accelerations
        linear_acc = total_force / self.mass

        # Angular acceleration
        torques = torques
        angular_acc = torques / self.I

        # Update state with semi-implicit Euler integration
        self.state[3:6] += linear_acc * self.dt
        self.state[9:12] += angular_acc * self.dt

        self.state[:3] += self.state[3:6] * self.dt
        self.state[6:9] += self.state[9:12] * self.dt

        # Normalize angles to [-pi, pi]
        self.state[6:9] = np.mod(self.state[6:9] + np.pi, 2 * np.pi) - np.pi

    def step(self, action):
        #self.render_mode = "human"

        body_rate_penalty_coeff = 0.0
        finish_reward = 50.0
        gate_reward = 10.0
        gate_passed_threshold = 0.5


        thrust = action[0]
        torques = action[1:]

        self._apply_dynamics(thrust, torques)
        current_gate = self.gates[self.current_gate_index]

        current_position = self.state[:3]
        body_rate = self.state[9:12]
        #print(current_position, "and", self.current_gate_index)


        # reward (from Scaramuzza)
        prev_distance_to_gate = np.linalg.norm(current_gate - self.prev_position)
        current_distance_to_gate = np.linalg.norm(current_gate - current_position)
        progress_reward = prev_distance_to_gate - current_distance_to_gate


        # body rate penalty
        body_rate_penalty = body_rate_penalty_coeff * np.linalg.norm(body_rate)

        reward = progress_reward - body_rate_penalty
        print(reward)

        done = False

        if current_distance_to_gate < gate_passed_threshold:
            reward += gate_reward
            self.current_gate_index += 1

            if self.current_gate_index == len(self.gates) - 1:  # All gates passed
                reward += finish_reward
                done = True


        z = self.state[2]
        angle_x = self.state[4]
        angle_y = self.state[5]
        angle_z = self.state[6]
        min_z, max_z = self.healthy_z_range
        min_angle, max_angle = self.healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = (
                min_angle < angle_x < max_angle and
                min_angle < angle_y < max_angle and
                min_angle < angle_z < max_angle
        )
        if not (healthy_z and healthy_angle):
            reward -= 0.0001


        observation = self._get_obs()

        info = {
            "position": current_position,
            "velocity": self.state[6:9],
            "current_gate": self.current_gate_index
        }

        if self.render_mode == "human":
            self.render()


        self.prev_position = current_position.copy()

        return observation, reward, done, False, info


    def _get_gate_corners(self, gate_centre):

        gate_width = 1.0
        gate_height = 1.0

        corners = [
            gate_centre + np.array([gate_width / 2, gate_height / 2, 0]),
            gate_centre + np.array([gate_width / 2, -gate_height / 2, 0]),
            gate_centre + np.array([-gate_width / 2, gate_height / 2, 0]),
            gate_centre + np.array([-gate_width / 2, -gate_height / 2, 0]),
        ]
        return corners


    def reset(self, seed=1, options=None):
        self.current_step = 0

        # Initialize state
        self.state = np.zeros(12, dtype=np.float32)

        self.prev_position = self.state[:3]

        observation = self._get_obs()

        info = {}
        return observation, info

    def set_design_params(self, design_params):
        self.design_params = design_params

    def set_env_id(self, env_id):
        self.env_id = env_id

    def get_design_params(self):
        return self.design_params

    def get_env_id(self):
        return self.env_id



