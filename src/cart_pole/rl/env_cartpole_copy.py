import numpy as np

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

from random import randrange
# import tf2_ros
# import tf.transformations as tf_trans
import sys
import os
sys.path.append('/home/divij/Documents/quadopter/devel/lib')

import cartpole_Model_wrapper as mw

class CartPoleEnv(utils.EzPickle, gym.Env):
    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 20,
    }

    def __init__(self, REWARD = np.array([0.0, 0.0, 0.0]), env_id = 0, pole_length = 1.0, call_back = f"constant_design"):
        super(CartPoleEnv, self).__init__()

        utils.EzPickle.__init__(self)
        self.call_back = call_back
        self.action_space = Box(low=-1, high=1, shape=(1,))
        if self.call_back is f"constant_design":
            self.observation_space = Box(low=-1, high=1, shape=(4,))
        else:
            self.observation_space = Box(low=-1, high=1, shape=(5,))
        self.model = mw.cart_pole_system_dynamics0()
        self.freq = 1
        self.steps = 0
        self.env_id = env_id
        #reward function
        self.TARGET_POLE_POS = REWARD[0]
        self.TARGET_CART_POS = REWARD[1]
        self.ACTION_PENALTY = REWARD[2]
        self.SURVIVAL_REWARD = 0.0
        self.CART_VEL_PENALTY = 0.0
        self.POLE_VEL_PENALTY = 0.0
        self.desired_cart_position = 0
        self.desired_pole_position = 0
        #Design parameters
        self.pole_length = pole_length
        self.MIN_POLE_LENGTH = 0.5
        self.MAX_POLE_LENGTH = 10
        self.cart_mass = 1
        self.pole_density = 0.5
        # self.pole_mass = 0.5
        self.pole_mass = self.pole_density * self.pole_length
        self.init_pole_pos = 0
        self.opt_change_len = False
        #action
        self.force_input = 0
        self.prev_action = 0
        self.MIN_FORCE = -10
        self.MAX_FORCE = 10

        #observation
        self.cart_pos = 0
        self.pole_angle = 0
        self.effort = 0
        self.cart_vel = 0
        self.pole_vel = 0
        self.pole_pos = np.array([0,0])

        self.MIN_CART_POS = -10
        self.MAX_CART_POS = 10

    
        np.set_printoptions(precision=2)

    def convert_range(self,x, min_x, max_x, min_y, max_y):
        return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y

    def step(self, action):
        self.steps += 1
        done = False
        early_Stop = False
        action = np.clip(action, -1, 1)
        reward_action = action
        action = self.convert_range(action ,-1, 1 ,self.MIN_FORCE, self.MAX_FORCE)
        
        
        self.model.set_length(self.pole_length)
        self.model.set_mass_cart(self.cart_mass)
        self.model.set_mass_pole(self.pole_mass)
        self.model.set_force_input_cart(action)
        
        for _ in range(self.freq):
            self.model.step()

        self.cart_pos = self.model.get_cart_position()
        obs_cart_pos = self.convert_range(self.cart_pos, self.MIN_CART_POS, self.MAX_CART_POS, -1, 1)
        theta = self.model.get_pole_angle()
        self.pole_angle = np.array([m.atan2(m.sin(theta), m.cos(theta))])
        self.effort = self.model.get_effort()
        self.pole_pos = self.model.get_pole_position()
        self.pole_vel = self.model.get_pole_angular_velocity()
        obs_pole_ang_vel = np.clip(self.pole_vel, -5, 5)
        self.cart_vel = self.model.get_cart_velocity()
        obs_cart_vel = np.clip(self.cart_vel, -10, 10)
        obs_cart_vel = self.convert_range(obs_cart_vel, -10, 10, -1, 1)
        obs_pole_length = np.array([self.convert_range(self.pole_length, self.MIN_POLE_LENGTH, self.MAX_POLE_LENGTH, -1, 1)])

        #reward function
        high_action_penalty_reward = -self.ACTION_PENALTY * m.fabs(reward_action)
        if self.cart_pos > self.MIN_CART_POS and self.cart_pos < self.MAX_CART_POS:
            desired_cart_position_reward = -m.fabs(obs_cart_pos-self.desired_cart_position) * self.TARGET_CART_POS
        else:
            desired_cart_position_reward = 0
        if abs(self.pole_angle) < m.pi:
            desired_pole_position_reward = m.cos(self.pole_angle) * self.TARGET_POLE_POS
        else:
            desired_pole_position_reward = 0
        # if self.env_id == 0:
        #     print("desired_pole_position_reward", desired_pole_position_reward)
        reward = high_action_penalty_reward + desired_cart_position_reward + desired_pole_position_reward

        if(self.cart_pos < self.MIN_CART_POS or self.cart_pos > self.MAX_CART_POS):
            reward = -1.0
            done = True
            early_Stop = True

        # if self.steps > 512 * 10 - 1:
        #     done = True
        #     early_Stop = True

        if self.call_back is f"constant_design":
            observation = np.concatenate([obs_cart_pos, obs_cart_vel, self.pole_angle, obs_pole_ang_vel])
        else:
            observation = np.concatenate([obs_cart_pos, obs_cart_vel, self.pole_angle, obs_pole_ang_vel, obs_pole_length])

        if (np.any(np.isnan(self.cart_pos)) or np.any(np.isinf(self.cart_pos)) or
            np.any(np.isnan(self.pole_angle)) or np.any(np.isinf(self.pole_angle)) or
            np.any(np.isnan(self.effort)) or np.any(np.isinf(self.effort)) or
            np.any(np.isnan(self.pole_pos)) or np.any(np.isinf(self.pole_pos))):
            print("nan or inf detected in reset")
            observation = np.ones(5)
            self.prev_action = action
            if self.call_back is f"constant_design":
                observation = np.ones(4)    
            else:
                observation = np.ones(5)
            reward = -1.0
            reward = float(reward)
            done = True
            early_Stop = True        

        self.prev_action = reward_action
        observation = observation.astype(np.float32)
        reward = float(reward)
        return observation, reward, done, early_Stop, {"cart_pos": self.cart_pos, "pole_pos":self.pole_pos, "pole_length":self.pole_length, "effort":self.effort, "pole_vel":self.pole_vel, "cart_vel":self.cart_vel, "pole_angle":self.pole_angle, "force":action}
    
    def reset(self, seed=None):
        self.steps = 0

        # if self.call_back is not f"constant_design":
        #     if not self.opt_change_len:
        #         self.pole_length = random.uniform(self.MIN_POLE_LENGTH, self.MAX_POLE_LENGTH)
        self.model.initialize()
        self.model.set_length(self.pole_length)
        obs_pole_length = np.array([self.convert_range(self.pole_length, self.MIN_POLE_LENGTH, self.MAX_POLE_LENGTH, -1, 1)])  
        self.model.set_mass_cart(self.cart_mass)
        self.pole_mass = self.pole_density * self.pole_length 
        self.model.set_mass_pole(self.pole_mass)
        sign = random.choice([-1, 1])
        init_pole_angle = sign * random.uniform(m.pi/2, m.pi)
        self.init_pole_pos = init_pole_angle
        self.model.set_init_pole_pos(self.init_pole_pos)

        self.cart_pos = self.model.get_cart_position()
        obs_cart_pos = self.convert_range(self.cart_pos, self.MIN_CART_POS, self.MAX_CART_POS, -1, 1)
        self.prev_action = 0
        self.pole_angle = self.model.get_pole_angle()
        self.effort = self.model.get_effort()
        self.pole_pos = self.model.get_pole_position()
        self.pole_vel = self.model.get_pole_angular_velocity()
        obs_pole_ang_vel = np.clip(self.pole_vel, -1, 1)
        self.cart_vel = self.model.get_cart_velocity()
        obs_cart_vel = np.clip(self.cart_vel, -10, 10)
        obs_cart_vel = self.convert_range(obs_cart_vel, -10, 10, -1, 1)

        if self.call_back is f"constant_design":
            observation = np.concatenate([obs_cart_pos, obs_cart_vel, self.pole_angle, obs_pole_ang_vel])
        else:
            observation = np.concatenate([obs_cart_pos, obs_cart_vel, self.pole_angle, obs_pole_ang_vel, obs_pole_length])

        if (np.any(np.isnan(self.cart_pos)) or np.any(np.isinf(self.cart_pos)) or
            np.any(np.isnan(self.pole_angle)) or np.any(np.isinf(self.pole_angle)) or
            np.any(np.isnan(self.effort)) or np.any(np.isinf(self.effort)) or
            np.any(np.isnan(self.pole_pos)) or np.any(np.isinf(self.pole_pos))):
            print("nan or inf detected in reset")
            if self.call_back is f"constant_design":
                observation = np.ones(4)    
            else:
                observation = np.ones(5)
            observation = observation.astype(np.float32)
            reward = -1.0  
            reward = float(reward) 

        return observation, {"cart_pos": self.cart_pos, "pole_pos":self.pole_pos}

    def set_pole_length(self, pole_length):
        self.pole_length = pole_length
        self.pole_mass = self.pole_density * self.pole_length 

        self.opt_change_len = True

    def get_pole_length(self):
        return self.pole_length
    
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