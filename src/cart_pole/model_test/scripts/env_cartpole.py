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

    def __init__(self, REWARD = np.array([1.0, 0.0])):
        super(CartPoleEnv, self).__init__()

        utils.EzPickle.__init__(self)
        self.action_space = Box(low=-1, high=1, shape=(1,))
        self.observation_space = Box(low=-1, high=1, shape=(5,))
        self.model = mw.cart_pole_system_dynamics0()
        self.freq = 1
        self.steps = 0
        #reward function
        self.TARGET_REWARD = REWARD[0]
        self.ACTION_PENALTY = REWARD[1]
        self.SURVIVAL_REWARD = 0.0
        self.CART_VEL_PENALTY = 0.0
        self.POLE_VEL_PENALTY = 0.0
        #Design parameters
        self.pole_length = 0.5
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.init_pole_pos = 0.01
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

        self.MIN_CART_POS = -4.8
        self.MAX_CART_POS = 4.8

    
        np.set_printoptions(precision=2)

    def convert_range(self,x, min_x, max_x, min_y, max_y):
        return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y

    def step(self, action):
        self.steps += 1
        done = False
        early_Stop = False
        reward_action = action               
        self.model.set_length(self.pole_length)
        self.model.set_mass_cart(self.cart_mass)
        self.model.set_mass_pole(self.pole_mass)
        self.model.set_force_input_cart(action)
        
        for _ in range(self.freq):
            self.model.step()

        self.cart_pos = self.model.get_cart_position()
        self.pole_angle = self.model.get_pole_angle()
        self.effort = self.model.get_effort()
        self.pole_pos = self.model.get_pole_position()
        self.pole_vel = self.model.get_pole_angular_velocity()
        self.cart_vel = self.model.get_cart_velocity()

        #reward function
        high_action_penalty = self.ACTION_PENALTY * m.fabs(reward_action) 
        reward = 1.0 - high_action_penalty

        if(self.cart_pos < self.MIN_CART_POS or self.cart_pos > self.MAX_CART_POS):
            reward = -1.0
            done = True
            early_Stop = True
    

        # observation = np.concatenate([self.cart_pos, self.cart_vel, self.rad_to_deg(self.pole_angle), self.rad_to_deg(self.pole_vel)])
        observation = np.concatenate([self.cart_pos, self.cart_vel, self.pole_angle, self.pole_vel, self.pole_pos])

        if (np.any(np.isnan(self.cart_pos)) or np.any(np.isinf(self.cart_pos)) or
            np.any(np.isnan(self.pole_angle)) or np.any(np.isinf(self.pole_angle)) or
            np.any(np.isnan(self.effort)) or np.any(np.isinf(self.effort)) or
            np.any(np.isnan(self.pole_pos)) or np.any(np.isinf(self.pole_pos))):
            print("nan or inf detected in reset")
            observation = np.ones(5)
            self.prev_action = action
            observation = observation.astype(np.float32)
            reward = -1.0
            reward = float(reward)
            done = True
            early_Stop = True        

        self.prev_action = reward_action
        observation = observation.astype(np.float32)
        reward = float(reward)
        return observation, reward, done, early_Stop, {"cart_pos": self.cart_pos, "pole_pos":self.pole_pos, "pole_length":self.pole_length, "effort":self.effort, "pole_vel":self.pole_vel, "cart_vel":self.cart_vel}
    
    def reset(self, seed=None):
        self.steps = 0
        self.model.initialize()


        self.model.set_length(self.pole_length)
        self.model.set_mass_cart(self.cart_mass)
        self.model.set_mass_pole(self.pole_mass)
        self.model.set_init_pole_pos(self.init_pole_pos)
        self.model.set_force_input_cart(0)
        self.model.step()

        self.cart_pos = self.model.get_cart_position()
        self.prev_action = 0
        self.pole_angle = self.model.get_pole_angle()
        self.effort = self.model.get_effort()
        self.pole_pos = self.model.get_pole_position()
        self.pole_vel = self.model.get_pole_angular_velocity()
        self.cart_vel = self.model.get_cart_velocity()
        # observation = np.concatenate([self.cart_pos, self.cart_vel, self.rad_to_deg(self.pole_angle), self.rad_to_deg(self.pole_vel)])
        observation = np.concatenate([self.cart_pos, self.cart_vel, self.pole_angle, self.pole_vel, self.pole_pos])

        if (np.any(np.isnan(self.cart_pos)) or np.any(np.isinf(self.cart_pos)) or
            np.any(np.isnan(self.pole_angle)) or np.any(np.isinf(self.pole_angle)) or
            np.any(np.isnan(self.effort)) or np.any(np.isinf(self.effort)) or
            np.any(np.isnan(self.pole_pos)) or np.any(np.isinf(self.pole_pos))):
            print("nan or inf detected in reset")
            observation = np.ones(5)
            observation = observation.astype(np.float32)
            reward = -1.0  
            reward = float(reward) 

        return observation, {"cart_pos": self.cart_pos, "pole_pos":self.pole_pos}

    def render(self, mode='human'):
        pass

    def close(self):
        pass   

    def deg_to_rad(self, deg):
        return deg * m.pi / 180.0
    
    def rad_to_deg(self, rad):
        return rad * 180.0 / m.pi