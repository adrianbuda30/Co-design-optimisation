import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.spaces import Box
import math as m
import control as ctrl
from scipy.optimize import minimize

from state_space_Rafa import state_space
import sys
import random

sys.path.append('/Users/adrianbuda/Downloads/master_thesis-aerofoil/devel/lib')

import aerofoil_wrapper

class LQREnv(utils.EzPickle, gym.Env):
    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 20,
    }

    def __init__(self, REWARD=np.array([0.0, 0.0]), env_id=0, mass=1.5, x_cg=0.62, x_ea=0.35, k_h=1800, k_alpha=370, k_b=390, call_back=f"Hebo_callback"):
        super(LQREnv, self).__init__()

        utils.EzPickle.__init__(self)

        self.call_back = call_back

        self.u_min = -10.0
        self.u_max = 10.0

        # reward function
        self.CL_PENALTY = REWARD[0]
        self.ACTION_PENALTY = REWARD[1]

        self.action_space = Box(low=self.u_min, high=self.u_max, shape=(1,))
        self.observation_space = Box(low=-1, high=1, shape=(6,))

        self.dT = 0.01
        self.Q = np.zeros((12, 12))  # state cost
        self.Q[0, 0] = 10
        self.Q[1, 1] = 10
        self.Q[2, 2] = 1
        self.Q[3, 3] = 1
        self.Q[4, 4] = 1
        self.Q[5, 5] = 1

        self.R = np.eye(1)  # control cost

        # Other initialization code...
        self.model = aerofoil_wrapper.aerofoil()  # needs to be changed
        self.freq = 1
        self.steps = 0
        self.env_id = env_id

        self.MatrixA = np.zeros(144)
        self.MatrixB = np.zeros(12)
        self.MatrixC = np.zeros(12)
        self.delta_ddot = 0

        self.checker = 0
        self.convergence_time = 0

        # Design parameters
        self.mass = mass
        self.MIN_MASS = 0.01
        self.MAX_MASS = 100

        self.x_cg = x_cg
        self.MIN_X_CG = 0.001
        self.MAX_X_CG = 0.8

        self.x_ea = x_ea
        self.MIN_X_EA = 0.001
        self.MAX_X_EA = 0.8

        self.k_h = k_h
        self.MIN_K_H = 0.01
        self.MAX_K_H = 5000

        self.k_alpha = k_alpha
        self.MIN_K_ALPHA = 0.01
        self.MAX_K_ALPHA = 5000

        self.k_b = k_b
        self.MIN_K_B = 0.01
        self.MAX_K_B = 5000

        # action
        self.delta_ddot = 0
        self.MIN_delta_ddot = -m.pi
        self.MAX_delta_ddot = m.pi

        # observation
        self.plunge = 0
        self.pitch = 0
        self.delta = 0
        self.plunge_dot = 0
        self.pitch_dot = 0
        self.delta_dot = 0
        self.CL = 0

        random_init_pitch = np.pi / 90 # np.random.uniform(-m.pi / 12, m.pi / 12)
        self.prev_states = np.zeros(12)
        self.prev_states[1] = random_init_pitch

        self.desired_pitch = np.array([0])


        self.MIN_PITCH = -m.pi / 3
        self.MAX_PITCH = m.pi / 3

        self.MIN_DELTA = -m.pi / 3
        self.MAX_DELTA = m.pi / 3

        self.MIN_PLUNGE = -1
        self.MAX_PLUNGE = 1

        self.time = 0
        self.consecutive_time = 0
        self.tol_prev = 0.0001
        self.tol_ref = 0.0002

        self.convergence_steps = 0

        np.set_printoptions(precision=5)

    def convert_range(self, x, min_x, max_x, min_y, max_y):
        return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y


    def set_design_params(self, new_design_params):
        self.mass = new_design_params[0]
        self.x_cg = new_design_params[1]
        self.x_ea = new_design_params[2]
        self.k_h = new_design_params[3]
        self.k_alpha = new_design_params[4]
        self.k_b = new_design_params[5]


    def step(self, action):
        self.steps += 1
        done = False
        early_Stop = False
        reward = 0
        reward1 = 0
        reward2 = 0
        # self.model.set_MatrixA(self.MatrixA)
        # self.model.set_MatrixB(self.MatrixB)
        # self.model.set_MatrixC(self.MatrixC)
        # action = self.convert_range(action, -m.pi, m.pi, -1, 1)
        # action = np.array([[0]])
        x = self.prev_states
        self.pitch_init = self.prev_states[0]
        self.plunge_init = self.prev_states[1]
        self.delta_init = self.prev_states[2]
        # self.CL = self.model.get_C_L()
        self.plunge_dot_init = self.prev_states[3]
        self.pitch_dot_init = self.prev_states[4]
        self.delta_dot_init = self.prev_states[5]
        #x = np.block([[self.plunge_init], [self.pitch_init], [self.delta_init], [self.plunge_dot_init], [self.pitch_dot_init], [self.delta_dot_init], [0], [0], [0], [0], [0], [0]])


        # optimal control
        u_optimal = -self.best_control_gains @ x

        dx = (self.A_ae - self.B_ae @ self.best_control_gains) @ x + self.n_bar * self.B_ae @ self.desired_pitch
        self.prev_states = self.prev_states + self.dT * dx

        self.plunge = self.prev_states[0]
        self.pitch = self.prev_states[1]
        self.delta = self.prev_states[2]
        # self.CL = self.model.get_C_L()
        self.plunge_dot = self.prev_states[3]
        self.pitch_dot = self.prev_states[4]
        self.delta_dot = self.prev_states[5]

        obs_plunge = self.plunge
        obs_plunge = np.clip(obs_plunge, -5, 5)
        obs_plunge = self.convert_range(obs_plunge, -5, 5, -1, 1)

        obs_pitch = self.pitch
        obs_delta = self.delta


        obs_pitch = np.clip(obs_pitch, -m.pi / 2, m.pi / 2)
        obs_pitch = self.convert_range(obs_pitch, -m.pi / 2, m.pi / 2, -1, 1)

        obs_delta = np.clip(obs_delta, -m.pi / 2, m.pi / 2)
        obs_delta = self.convert_range(obs_delta, -m.pi / 2, m.pi / 2, -1, 1)

        obs_plunge_dot = self.plunge_dot
        obs_plunge_dot = np.clip(obs_plunge_dot, -5, 5)
        obs_plunge_dot = self.convert_range(obs_plunge_dot, -5, 5, -1, 1)

        obs_pitch_dot = self.pitch_dot
        obs_pitch_dot = np.clip(obs_pitch_dot, -m.pi / 2, m.pi / 2)
        obs_pitch_dot = self.convert_range(obs_pitch_dot, -m.pi / 2, m.pi / 2, -1, 1)

        obs_delta_dot = self.delta_dot
        obs_delta_dot = np.clip(obs_delta_dot, -m.pi / 2, m.pi / 2)
        obs_delta_dot = self.convert_range(obs_delta_dot, -m.pi / 2, m.pi / 2, -1, 1)

        self.state = np.array(
            [[self.plunge], [self.pitch], [self.delta], [self.plunge_dot], [self.pitch_dot], [self.delta_dot], [0], [0], [0], [0], [0], [0]])

        self.CL = self.MatrixC @ self.state

        obs_CL = self.CL
        obs_CL = np.clip(obs_CL, -10, 10)
        obs_CL = self.convert_range(obs_CL, -10, 10, -1, 1)

        # reward function

        # reward function

        self.time += self.dT

        if abs(self.pitch - self.desired_pitch) < self.tol_ref:
            self.consecutive_time += self.dT
        else:
            self.consecutive_time = 0



        observation = np.concatenate(
            [[obs_plunge], [obs_pitch], [obs_delta], [obs_plunge_dot], [obs_pitch_dot], [obs_delta_dot]])


        #reward = 1 / (1 + abs(self.pitch - self.desired_pitch) * 180 / np.pi)
        reward = 1 / (1 + (self.pitch * 180 / np.pi) ** 2 * 10 + (u_optimal  * 180 / np.pi) ** 2)
        if (self.consecutive_time > 20 or self.time >= 100) and self.checker == 0:
            self.convergence_time = self.time
            reward2 = - self.convergence_time / 100
            print(self.convergence_time)
            self.checker = 1

        if self.checker == 1:
            self.convergence_steps = 1


        #if self.pitch < self.MIN_PITCH or self.pitch > self.MAX_PITCH or self.delta < self.MIN_DELTA or self.delta > self.MAX_DELTA or self.plunge < self.MIN_PLUNGE or self.plunge > self.MAX_PLUNGE:
        #    reward = -1.0
        #    early_Stop = False

        if (np.any(np.isnan(observation)) or np.any(np.isinf(observation))):
            # print("nan or inf detected in reset")
            #print("Observations:", observation)
            #print("Reward:", reward)
            observation = np.ones(7)
            reward = - 5000.0
            reward = float(reward)
            done = True
            early_Stop = True


        observation = observation.astype(np.float32)
        reward = float(reward)
        #print(reward)
        info = {"pitch": self.pitch, "plunge": self.plunge, "delta": self.delta,
                "CL": self.CL, "delta_ddot_input": action, "MatrixA": self.MatrixA,
                "MatrixB": self.MatrixB, "MatrixC": self.MatrixC,
                "plunge_dot": self.plunge_dot, "pitch_dot": self.pitch_dot,
                "delta_dot": self.delta_dot, "convergence_time": self.convergence_time, "convergence_steps": self.convergence_steps}
        return observation, reward, done, early_Stop, info

    def reset(self, seed=None):
        self.steps = 0
        self.time = 0

        self.convergence_time = 0
        self.convergence_steps = 0

        #if self.call_back is not f"constant_design":
        #    self.mass = 1.5 #random.uniform(self.MIN_MASS, self.MAX_MASS)
        #    self.x_cg = 0.159 #random.uniform(self.MIN_X_CG, self.MAX_X_CG)
        #    self.x_ea = 0.364 #random.uniform(self.MIN_X_EA, self.MAX_X_EA)
        #    self.k_h = 356.83 #random.uniform(self.MIN_K_H, self.MAX_K_H)
        #    self.k_alpha = 784.91 #random.uniform(self.MIN_K_ALPHA, self.MAX_K_ALPHA)
        #    self.k_b = 0 #random.uniform(self.MIN_K_B, self.MAX_K_B)

        self.MatrixA, self.MatrixB, self.MatrixC = state_space(self.mass, self.x_cg, self.x_ea, self.k_h,
                                                               self.k_alpha, self.k_b)
        self.n = self.MatrixA.shape[0]  # states
        self.m = self.MatrixB.shape[1]  # inputs

        K, S, E = ctrl.lqr(self.MatrixA, self.MatrixB, self.Q, self.R)
        self.best_control_gains = K.reshape((1, 12))


        Z = np.block([[np.zeros((1, self.n)), 1]]);
        N = np.linalg.inv(np.block([[self.MatrixA, self.MatrixB],
                                    [self.MatrixC, 0]])) @ np.transpose(Z)
        Nx = N[0:self.n]
        Nu = N[self.n]

        self.n_bar = Nu + K @ Nx
        self.n_bar = self.n_bar[0, 0]


        #print(self.MatrixA)
        #print(self.MatrixB)
        #print(self.MatrixC)
        # Matrix A needs to be first transposed and then flattened

        self.A_ae = self.MatrixA
        self.B_ae = self.MatrixB
        self.C_ae = self.MatrixC

        self.MatrixA = self.MatrixA.T
        self.MatrixA = self.MatrixA.flatten()
        self.MatrixB = self.MatrixB.flatten()
        self.MatrixC = self.MatrixC.flatten()

        self.model.set_MatrixA(self.MatrixA)
        self.model.set_MatrixB(self.MatrixB)
        self.model.set_MatrixC(self.MatrixC)
        random_init_pitch = np.pi / 90 # np.random.uniform(-m.pi / 12, m.pi / 12)
        self.init_state = np.zeros(12)
        self.init_state[1] = random_init_pitch
        self.prev_states = self.init_state
        self.model.set_init_state(self.init_state)
        self.model.set_delta_ddot(self.delta_ddot)

        self.checker = 0
        self.consecutive_time = 0

        self.model.initialize()
        # self.model.step()

        self.plunge = self.prev_states[0]
        self.pitch = self.prev_states[1]
        self.delta = self.prev_states[2]
        # self.CL = self.model.get_C_L()
        self.plunge_dot = self.prev_states[3]
        self.pitch_dot = self.prev_states[4]
        self.delta_dot = self.prev_states[5]


        obs_plunge = self.plunge
        obs_plunge = np.clip(obs_plunge, -5, 5)
        obs_plunge = self.convert_range(obs_plunge, -5, 5, -1, 1)

        obs_pitch = self.pitch# % 360
        obs_delta = self.delta# % 360

        #self.CL = 2 * np.sin(obs_pitch) * np.cos(obs_pitch) + 2 * np.sin(obs_delta) * np.cos(obs_delta)

        obs_pitch = np.clip(obs_pitch, -m.pi / 2, m.pi / 2)
        obs_pitch = self.convert_range(obs_pitch, -m.pi / 2, m.pi / 2, -1, 1)

        obs_delta = np.clip(obs_delta, -m.pi / 2, m.pi / 2)
        obs_delta = self.convert_range(obs_delta, -m.pi / 2, m.pi / 2, -1, 1)


        obs_plunge_dot = self.plunge_dot
        obs_plunge_dot = np.clip(obs_plunge_dot, -5, 5)
        obs_plunge_dot = self.convert_range(obs_plunge_dot, -5, 5, -1, 1)

        obs_pitch_dot = self.pitch_dot
        obs_pitch_dot = np.clip(obs_pitch_dot, -m.pi / 2, m.pi / 2)
        obs_pitch_dot = self.convert_range(obs_pitch_dot, -m.pi / 2, m.pi / 2, -1, 1)

        obs_delta_dot = self.delta_dot
        obs_delta_dot = np.clip(obs_delta_dot, -m.pi / 2, m.pi / 2)
        obs_delta_dot = self.convert_range(obs_delta_dot, -m.pi / 2, m.pi / 2, -1, 1)

        self.state = np.concatenate(
            [[self.plunge],[self.pitch], [self.delta], [self.plunge_dot], [self.pitch_dot], [self.delta_dot], [0], [0], [0], [0], [0], [0]])

        self.CL = self.MatrixC @ self.state
        obs_CL = self.CL

        obs_CL = np.clip(obs_CL, -5, 5)
        obs_CL = self.convert_range(obs_CL, -5, 5, -1, 1)

        observation = np.concatenate(
            [[obs_plunge], [obs_pitch], [obs_delta], [obs_plunge_dot], [obs_pitch_dot], [obs_delta_dot]])

        if (np.any(np.isnan(observation)) or np.any(np.isinf(observation))):
            #print("nan or inf detected in reset")
            #print("Observations:", observation)
            # print("Reward:", reward)
            observation = np.ones(7)
            observation = observation.astype(np.float32)
            reward = - 1000.0
            reward = float(reward)

        observation = observation.astype(np.float32)

        info = {"pitch": self.pitch, "plunge": self.plunge, "delta": self.delta,
                "CL": self.CL, "delta_ddot_input": self.delta_ddot, "MatrixA": self.MatrixA,
                "MatrixB": self.MatrixB, "MatrixC": self.MatrixC,
                "plunge_dot": self.plunge_dot, "pitch_dot": self.pitch_dot,
                "delta_dot": self.delta_dot, "convergence_time": self.convergence_time}

        return observation, info


    def get_design_params(self):
        design_params = np.array([self.mass, self.x_cg, self.x_ea, self.k_h, self.k_alpha])
        return design_params

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

