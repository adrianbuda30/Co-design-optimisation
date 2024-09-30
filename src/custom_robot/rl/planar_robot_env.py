import gymnasium as gym
from gymnasium import spaces
from gymnasium import utils
from gymnasium.spaces import Box
import math as m
import random
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import sys
import math as m
import time
from scipy.io import loadmat, savemat

sys.path.append('/home/divij/Documents/quadopter/devel/lib')
import custom_robot_wrapper

class PlanarRobotEnv(utils.EzPickle, gym.Env):
    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 20,
    }

    def __init__(self, REWARD = np.array([0.0, 0.0, 0.0]), env_id = 0, arm_length = np.array([3.0, 2.0, 1.0]), call_back = f"random_design"):
        super(PlanarRobotEnv, self).__init__()

        utils.EzPickle.__init__(self)
        self.call_back = call_back
        self.action_space = Box(low=-1, high=1, shape=(1,))

        self.observation_space = Box(low=-1, high=1, shape=(5,))
        self.model = custom_robot_wrapper
        self.freq = 1
        self.steps = 0
        self.env_id = env_id
        #reward function
        self.TARGET_POLE_POS = REWARD[0]
        self.TARGET_CART_POS = REWARD[1]
        self.ACTION_PENALTY = REWARD[2]

        # Define input parameters
        self.curr_joint_pos = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.curr_joint_vel = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.rho = np.array(1000.0, dtype=np.double)
        self.radius = np.array(0.02, dtype=np.double)
        self.arm_length = np.array(arm_length, dtype=np.double)

        # Initialize output arrays
        self.pos_tcp = np.empty(3, dtype=np.double)
        self.curr_joint_acc = np.array([0.0, 0.0, 0.0], dtype=np.double)

        # Initial time and time step
        self.dt = 0.001 # 1 ms

        self.max_length = np.array([3.0, 2.0, 1.0], dtype=np.double)
        self.min_length = np.array([0.5, 0.5, 0.5], dtype=np.double)
        self.max_joint_pos = np.array([m.pi, m.pi, m.pi], dtype=np.double)
        self.min_joint_pos = np.array([-m.pi, -m.pi, -m.pi], dtype=np.double)
        self.min_joint_vel = np.array([-100.0, -100.0, -100.0], dtype=np.double)
        self.max_joint_vel = np.array([100.0, 100.0, 100.0], dtype=np.double)
        self.max_torque = np.array([100.0, 100.0, 100.0], dtype=np.double)
        self.min_torque = np.array([-100.0, -100.0, -100.0], dtype=np.double)
        self.pos_tcp_range_x = np.array([0.0, 0.0], dtype=np.double)
        self.tcp_pos_range_y = np.array([0.0, 0.0], dtype=np.double)
        # Instantiate the FirstOrderDelay class
        self.filter = FirstOrderDelay(alpha=0.1)

        np.set_printoptions(precision=2)

    def convert_range(self,x, min_x, max_x, min_y, max_y):
        return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y

    def step(self, action):

        self.steps += 1
        joint_torque_real = self.convert_range(action, -1, 1, self.min_torque, self.max_torque)
        self.joint_torque = np.array(self.filter.step(joint_torque_real), dtype=np.double)
        
        # Call the wrapped function
        self.model.calc_sys_matrices(self.curr_joint_pos, self.curr_joint_vel, 
                                    self.rho, self.radius, self.arm_length, 
                                    self.joint_torque, self.curr_joint_acc, self.pos_tcp)

        # Integrating the acceleration to get velocity
        self.curr_joint_vel = self.curr_joint_vel + self.curr_joint_acc * self.dt
        self.curr_joint_vel = np.array(self.curr_joint_vel, dtype=np.double)
        # Integrating the velocity to get position
        self.curr_joint_pos = self.curr_joint_pos + self.curr_joint_vel * self.dt
        self.curr_joint_pos = np.array(self.curr_joint_pos, dtype=np.double)

        obs_joint_torque = self.convert_range(self.joint_torque, self.min_torque, self.max_torque, -1, 1)
        obs_joint_pos = self.convert_range(self.curr_joint_pos, self.min_joint_pos, self.max_joint_pos, -1, 1)
        obs_joint_vel = self.convert_range(self.curr_joint_vel, self.min_joint_vel, self.max_joint_vel, -1, 1)
        obs_arm_length = self.convert_range(self.arm_length, self.min_length, self.max_length, -1, 1)
        obs_joint_acc = self.convert_range(self.curr_joint_acc, self.min_joint_vel, self.max_joint_vel, -1, 1)
        obs_tcp_pos = self.convert_range(self.pos_tcp, self.pos_tcp_range_x[0], self.pos_tcp_range_x[1], -1, 1)

        observation = np.concatenate([obs_joint_torque, obs_tcp_pos, obs_joint_pos,
                                    obs_joint_vel, obs_joint_acc, obs_arm_length])

        if (np.any(np.isnan(obs_tcp_pos)) or np.any(np.isinf(obs_tcp_pos)) or
            np.any(np.isnan(obs_joint_acc)) or np.any(np.isinf(obs_joint_acc)) or
            np.any(np.isnan(obs_joint_torque)) or np.any(np.isinf(obs_joint_torque))):

            print("nan or inf detected in reset")
            observation = np.ones(5)
            observation = np.ones(5)
            reward = -1.0
            reward = float(reward)
            done = True
            early_Stop = True        

        observation = observation.astype(np.float32)
        reward = float(reward)
        info = {}
        return observation, reward, done, early_Stop, info
    
    def reset(self, seed=None):

        self.steps = 0

        # reset the robot

        self.curr_joint_pos = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.curr_joint_vel = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.curr_joint_acc = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.pos_tcp = np.empty(3, dtype=np.double)
        self.joint_torque = np.array([0.0, 0.0, 0.0], dtype=np.double)

        # randomize the arm length
        if self.call_back == "random_design":
            self.arm_length = np.array([random.uniform(self.min_length, self.max_length), 
                                        random.uniform(self.min_length, self.max_length), 
                                        random.uniform(self.min_length, self.max_length)], dtype=np.double)
        
        obs_joint_torque = self.convert_range(self.joint_torque, self.min_torque, self.max_torque, -1, 1)
        obs_joint_pos = self.convert_range(self.curr_joint_pos, self.min_joint_pos, self.max_joint_pos, -1, 1)
        obs_joint_vel = self.convert_range(self.curr_joint_vel, self.min_joint_vel, self.max_joint_vel, -1, 1)
        obs_arm_length = self.convert_range(self.arm_length, self.min_length, self.max_length, -1, 1)
        obs_joint_acc = self.convert_range(self.curr_joint_acc, self.min_joint_vel, self.max_joint_vel, -1, 1)
        obs_tcp_pos = self.convert_range(self.pos_tcp, self.pos_tcp_range_x[0], self.pos_tcp_range_x[1], -1, 1)

        observation = np.concatenate([obs_joint_torque, obs_tcp_pos, obs_joint_pos,
                                    obs_joint_vel, obs_joint_acc, obs_arm_length])
        
        if (np.any(np.isnan(obs_tcp_pos)) or np.any(np.isinf(obs_tcp_pos)) or
            np.any(np.isnan(obs_joint_acc)) or np.any(np.isinf(obs_joint_acc)) or
            np.any(np.isnan(obs_joint_torque)) or np.any(np.isinf(obs_joint_torque))):

            print("nan or inf detected in reset")
            observation = np.ones(5)
            observation = np.ones(5)
            reward = -1.0
            reward = float(reward)
            done = True
            early_Stop = True     

        info = {}
        return observation, info

    def set_arm_length(self, arm_length):
        self.arm_length = arm_length

    def get_arm_length(self):
        return self.arm_length
    
    def set_env_id(self, env_id):
        self.env_id = env_id
    
    def get_env_id(self):
        return self.env_id

    def render(self, mode='human'):
        pass

    def close(self):
        pass   

    def deg_to_rad(self, deg):
        return deg * m.pi / 180.0
    
class FirstOrderDelay:
    def __init__(self, alpha, initial_output=0):
        self.alpha = alpha
        self.previous_output = initial_output

    def step(self, current_input):
        current_output = (1 - self.alpha) * self.previous_output + self.alpha * current_input
        self.previous_output = current_output
        return current_output

