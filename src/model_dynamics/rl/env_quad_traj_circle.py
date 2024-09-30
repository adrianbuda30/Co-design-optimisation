import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.spaces import Box
import math as m
import random
import sys
from quadcopter_trajectory import QuadcopterTrajectory as trajectory
sys.path.append('/home/divij/Documents/quadopter/devel/lib')

import Model_wrapper as mw

class QuadcopterEnv(utils.EzPickle, gym.Env):
    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 20,
    }

    def __init__(self, arm_length = np.array([0.17,0.17,0.17,0.17]), REWARD = np.array([0.0, 0.5, 0.0]), env_id = 0, arm_length_limits = np.array([0.01, 2.0]) ,n_steps_train = 512 * 10):
        super(QuadcopterEnv, self).__init__()

        utils.EzPickle.__init__(self)
        self.action_space = Box(low=-1, high=1, shape=(4,))

        self.Max_propeller_speed = 4000
        self.Min_propeller_speed = 0
        self.n_steps_train = n_steps_train
        self.traj_length = n_steps_train
        self.arm_length_limits = arm_length_limits

        self.freq = 4
        self.env_id = env_id
        self.observation_space = Box(low=-1, high=1, shape=(31,))
        self.steps = 0
        self.init_num = 0
        self.model = mw.multirotor0()
        self.TARGET_POS = np.array([0.0, 0.0, 0.0])  # for example
        self.QUAT_DES = np.array([1.0, 0.0, 0.0, 0.0])  # for example
        self.R_DES = np.array([1.0, 0.0 , 0.0 , 0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 1.0])
        self.INIT_POS = np.array([0.0, 0.0, 0.0])  # for example
        self.INIT_VEL = np.array([0.0, 0.0, 0.0])  # for example
        self.INIT_OMEGA = np.array([0, 0.0, 0.0])  # for example

        self.max_init_vel = 0
        self.max_init_omega = 0
        self.MAX_POS = 5
        self.MAX_OMEGA = 5.0
        self.MAX_VEL = 10.0
        self.major_radius = 2.0
        self.minor_radius = 0.5

        #reward function
        self.SURVIVAL_REWARD = REWARD[0]
        self.VELOCITY_REWARD = REWARD[1]
        self.ANGULAR_VELOCITY_PENALTY = -REWARD[2]

        # #Design parameters
        # self.arm_length = np.array([0.291359,
        #                             0.496038,
        #                             0.812976,
        #                             0.460921])
        #Design parameters
        self.arm_length = np.array([0.2562,
                                    0.4945,
                                    0.8431,
                                    0.5090])
        # self.arm_length = np.array([0.213243,
        #                             0.358400,
        #                             1.188221,
        #                             0.436289])

        self.propeller_height = np.array([-0.028, -0.028, -0.028, -0.028])
        self.propeller_diameter = np.array([0.2032, 0.2032, 0.2032, 0.2032])
        self.rotation_direction = np.array([-1, 1, -1, 1])
        self.max_rpm = np.array([self.Max_propeller_speed, self.Max_propeller_speed, self.Max_propeller_speed, self.Max_propeller_speed])
        self.min_rpm = np.array([0.0, 0.0, 0.0, 0.0])
        self.arm_radius = np.array([0.0013, 0.0013, 0.0013, 0.0013])
        self.motor_arm_angle = np.array([45, 135, 225, 315])
        self.mass_center = 0.1
        self.COM_mass_center = np.array([0, 0, 0])
        self.set_Surface_params = np.array([0.12, 0.12, 0.06])

        self.opt_change_design_par = False
        np.set_printoptions(precision=2)

    def step(self, action):
        self.steps += 1
        done = False
        if action is np.inf or action is -np.inf:
            print("action is inf")
        w  = self.convert_range(action, -1, 1, self.Min_propeller_speed, self.Max_propeller_speed)

        #set design parameters and the rpm
        self.model.set_arm_length(self.arm_length)
        self.model.set_propeller_height(self.propeller_height)
        self.model.set_propeller_diameter(self.propeller_diameter)
        self.model.set_rotation_direction(self.rotation_direction)
        self.model.set_max_rpm(self.max_rpm)
        self.model.set_min_rpm(self.min_rpm)
        self.model.set_arm_radius(self.arm_radius)
        self.model.set_motor_arm_angle(self.motor_arm_angle)
        self.model.set_mass_center(self.mass_center)
        self.model.set_COM_mass_center(self.COM_mass_center)
        self.model.set_Surface_params(self.set_Surface_params)

        self.model.set_wind_vector(np.array([0.0 * np.sin(0.2 * self.steps/100),
                                            0.0 * np.cos(0.3 * self.steps/100),
                                            0.0 * np.sin(0.1 * self.steps/100)]))

        self.model.set_force_disturb_vector(np.array([0.0, 0.0, 0.0]))
        self.model.set_moment_disturb_vector(np.array([0.0, 0.0, 0.0]))
        self.model.set_w_0(w)
    
        self.model.step2() #sets the propeller speeds
        #step the model at self.freq times
        for i in range(self.freq):
            self.model.step0()

        #get position    
        pos_world = self.model.get_pos_world()
        obs_pos_world_norm = np.clip(pos_world, -self.MAX_POS, self.MAX_POS)
        obs_pos_world_norm = self.convert_range(pos_world, -self.MAX_POS, self.MAX_POS, -1, 1)

        #get rotation matrix
        R = self.model.get_RotationMatrix_world()
        obs_rotation_matrix = np.clip(R, -1, 1)
        obs_rotation_matrix = obs_rotation_matrix.flatten()

        #get velocity
        vel_world = self.model.get_velocity_world()
        obs_velocity_vector = vel_world / np.linalg.norm(vel_world)
        obs_velocity_des_grad = self.gradient_direction(pos_world[0], pos_world[1])
        obs_velocity_dot_product = np.dot(obs_velocity_vector, obs_velocity_des_grad)
        obs_vel_mag = np.linalg.norm(vel_world)

        #get angular velocity
        omega_world = self.model.get_omega_world()
        omega_world = np.clip(omega_world, -self.MAX_OMEGA, self.MAX_OMEGA)
        obs_omega_norm = self.convert_range(omega_world, -self.MAX_OMEGA, self.MAX_OMEGA, -1, 1)

        #clip arm length
        obs_arm_length = np.clip(self.arm_length, self.arm_length_limits[0], self.arm_length_limits[1])
        obs_arm_length = self.convert_range(obs_arm_length, self.arm_length_limits[0], self.arm_length_limits[1], -1, 1)

        #get propeller speed
        propeller_speed = self.model.get_motor_rpm()
        propeller_speed = np.clip(propeller_speed, self.Min_propeller_speed, self.Max_propeller_speed)
        obs_propeller_speed_norm = self.convert_range(propeller_speed, self.Min_propeller_speed, self.Max_propeller_speed, -1, 1)

        #reward function
        reward_vel_dot_des_grad = np.linalg.norm(vel_world) * obs_velocity_dot_product
        reward_vel_dot_des_grad = self.convert_range(reward_vel_dot_des_grad, -self.MAX_VEL, self.MAX_VEL, 0, 1)

        reward_ang_vel = np.linalg.norm(omega_world)
        reward_ang_vel_norm = self.convert_range(reward_ang_vel, -self.MAX_OMEGA, self.MAX_OMEGA, 0, 1)
        reward_ang_vel_norm = np.clip(reward_ang_vel_norm, 0, 1)

        reward_survival = self.convert_range(self.steps, 0, self.traj_length, 0, 1) 

        reward = self.SURVIVAL_REWARD * reward_survival + self.VELOCITY_REWARD * reward_vel_dot_des_grad + self.ANGULAR_VELOCITY_PENALTY * reward_ang_vel_norm

        if not(self.is_inside_torus(pos_world[0], pos_world[1], pos_world[2], self.major_radius, self.minor_radius)):
            done = True
            reward = -1.0

        if (np.any(np.isnan(pos_world)) or np.any(np.isinf(pos_world)) or
            np.any(np.isnan(vel_world)) or np.any(np.isinf(vel_world)) or
            np.any(np.isnan(omega_world)) or np.any(np.isinf(omega_world)) or
            np.any(np.isnan(R)) or np.any(np.isinf(R))):
            print("nan or inf detected")
            observation = np.ones(31)
            observation = observation.astype(np.float32)
            reward = -1.0
            reward = float(reward)
            early_Stop = False
            done  = True
            return observation, reward, done, early_Stop, {"pos_world": pos_world, "target_pos": self.TARGET_POS, "init_pos": self.INIT_POS, "rot_matrix_world":R.flatten(), "vel_world": vel_world, "omega_world": omega_world, "action": action, "reward": reward, "propeller_speed": propeller_speed, "action_motor_rpm": w, "arm_length": self.arm_length, "action_penalty": 0.0, "omega_world_penalty": 0.0, "steps": self.steps}
        
        observation = np.concatenate([obs_velocity_vector, 
                                      obs_velocity_des_grad,
                                      np.array([obs_velocity_dot_product]),
                                      np.array([obs_vel_mag]),
                                      obs_pos_world_norm,
                                      obs_omega_norm,
                                      obs_rotation_matrix, 
                                      obs_propeller_speed_norm,
                                      obs_arm_length])

        observation = observation.astype(np.float32)
        reward = float(reward)
        early_Stop = False
        return observation, reward, done, early_Stop, {"pos_world": pos_world, "init_pos": self.INIT_POS, "rot_matrix_world":R.flatten(), "vel_world": vel_world, "omega_world": omega_world, "action": action, "reward": reward, "propeller_speed": propeller_speed, "action_motor_rpm": w, "arm_length": self.arm_length, "steps": self.steps}
    
    def reset(self, seed=None):
        
        self.steps = 0
        # if not self.opt_change_design_par:
        #     self.arm_length = np.array([random.uniform(self.arm_length_limits[0],self.arm_length_limits[1]),
        #                                 random.uniform(self.arm_length_limits[0],self.arm_length_limits[1]), 
        #                                 random.uniform(self.arm_length_limits[0],self.arm_length_limits[1]), 
        #                                 random.uniform(self.arm_length_limits[0],self.arm_length_limits[1])])

        self.set_Surface_params = np.array([(self.arm_length[0]+self.arm_length[2])/2, 
                                            (self.arm_length[1]+self.arm_length[3])/2, 
                                            0.06])

            # self.motor_arm_angle = np.array([random.uniform(0, 90), random.uniform(90, 180), random.uniform(180, 270), random.uniform(270, 360)])
        self.INIT_POS = self.random_point_on_circle(self.major_radius)
        self.INIT_VEL = np.array(self.gradient_direction(self.INIT_POS[0], self.INIT_POS[1]))
        self.INIT_OMEGA = np.array([random.uniform(-self.max_init_omega, self.max_init_omega), random.uniform(-self.max_init_omega, self.max_init_omega),random.uniform(-self.max_init_omega, self.max_init_omega)], dtype=np.float64)
        
        #set design parameters and the rpm
        self.model.set_w_0(np.array([0.0, 0.0, 0.0, 0.0]))
        self.model.set_arm_length(self.arm_length)
        self.model.set_propeller_height(self.propeller_height)
        self.model.set_propeller_diameter(self.propeller_diameter)
        self.model.set_rotation_direction(self.rotation_direction)
        self.model.set_max_rpm(self.max_rpm)
        self.model.set_min_rpm(self.min_rpm)
        self.model.set_arm_radius(self.arm_radius)
        self.model.set_motor_arm_angle(self.motor_arm_angle)
        self.model.set_init_pos(self.INIT_POS)
        self.model.set_init_vel(self.INIT_VEL)
        self.model.set_init_omega(self.INIT_OMEGA)
        self.model.set_Surface_params(self.set_Surface_params)

        self.model.initialize()
        self.model.step2() #sets the propeller speeds
        self.model.step0()
        
        #get position    
        pos_world = self.model.get_pos_world()
        obs_pos_world_norm = np.clip(pos_world, -self.MAX_POS, self.MAX_POS)
        obs_pos_world_norm = self.convert_range(pos_world, -self.MAX_POS, self.MAX_POS, -1, 1)

        #get rotation matrix
        R = self.model.get_RotationMatrix_world()
        obs_rotation_matrix = np.clip(R, -1, 1)
        obs_rotation_matrix = obs_rotation_matrix.flatten()

        #get velocity
        vel_world = self.model.get_velocity_world()
        obs_velocity_vector = vel_world / np.linalg.norm(vel_world)
        obs_velocity_des_grad = self.gradient_direction(pos_world[0], pos_world[1])
        obs_velocity_dot_product = np.dot(obs_velocity_vector, obs_velocity_des_grad)
        obs_vel_mag = np.linalg.norm(vel_world)

        #get angular velocity
        omega_world = self.model.get_omega_world()
        omega_world = np.clip(omega_world, -self.MAX_OMEGA, self.MAX_OMEGA)
        obs_omega_norm = self.convert_range(omega_world, -self.MAX_OMEGA, self.MAX_OMEGA, -1, 1)

        #clip arm length
        obs_arm_length = np.clip(self.arm_length, self.arm_length_limits[0], self.arm_length_limits[1])
        obs_arm_length = self.convert_range(obs_arm_length, self.arm_length_limits[0], self.arm_length_limits[1], -1, 1)

        #get propeller speed
        propeller_speed = self.model.get_motor_rpm()
        propeller_speed = np.clip(propeller_speed, self.Min_propeller_speed, self.Max_propeller_speed)
        obs_propeller_speed_norm = self.convert_range(propeller_speed, self.Min_propeller_speed, self.Max_propeller_speed, -1, 1)

        observation = np.concatenate([obs_velocity_vector, 
                                      obs_velocity_des_grad,
                                      np.array([obs_velocity_dot_product]),
                                      np.array([obs_vel_mag]),
                                      obs_pos_world_norm,
                                      obs_omega_norm,
                                      obs_rotation_matrix, 
                                      obs_propeller_speed_norm,
                                      obs_arm_length])
        
        observation = observation.astype(np.float32)
        info = {}

        if (np.any(np.isnan(pos_world)) or np.any(np.isinf(pos_world)) or
            np.any(np.isnan(vel_world)) or np.any(np.isinf(vel_world)) or
            np.any(np.isnan(omega_world)) or np.any(np.isinf(omega_world)) or
            np.any(np.isnan(R)) or np.any(np.isinf(R))):
            print("nan or inf detected in reset")
            print(observation)
            observation = np.ones(31)
            observation = observation.astype(np.float32)
            reward = -1.0
            reward = float(reward)

        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_arm_length(self):
        return self.arm_length
    
    def set_arm_length(self, new_arm_length):
        self.arm_length = new_arm_length
        self.opt_change_design_par = True
  
    def is_within_radius(self,desired_point, current_point, radius):
        distance = np.linalg.norm(desired_point - current_point)
        return distance <= radius

    def set_env_id(self, env_id):
        self.env_id = env_id
    
    def get_env_id(self):
        return self.env_id

    def rotation_matrix_to_quaternion(self, rotation_matrix: np.array) -> tuple:
        a, b, c = rotation_matrix[0]
        d, e, f = rotation_matrix[1]
        g, h, i = rotation_matrix[2]
        q0 = np.sqrt(max(0, 1 + a + e + i)) / 2
        q1 = np.sqrt(max(0, 1 + a - e - i)) / 2
        q2 = np.sqrt(max(0, 1 - a + e - i)) / 2
        q3 = np.sqrt(max(0, 1 - a - e + i)) / 2
        q1 *= np.sign(f - h)
        q2 *= np.sign(g - c)
        q3 *= np.sign(b - d)
        magnitude = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        return q0 / magnitude, q1 / magnitude, q2 / magnitude, q3 / magnitude
    
    def cartesian_to_spherical_with_orientation(self, point, origin, R):
        # 1. Translate the point to make origin as (0, 0, 0)
        translated_point = point - origin

        # 2. Convert the translated point to spherical coordinates
        x, y, z = translated_point
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # 3. Convert the spherical coordinates' angles with orientation in mind
        rotated_point = np.dot(R.reshape(3,3), translated_point)
        x_rot, y_rot, z_rot = rotated_point
        theta = np.arccos(z_rot / r) if r != 0 else 0
        phi = np.arctan2(y_rot, x_rot)
        
        return r, theta, phi
    
    def convert_range(self,x, min_x, max_x, min_y, max_y):
        return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y
    
    def is_traj_in_bounds(self, points, bound):

        points = np.array(points)
        is_in_bounds = not (np.any(points>bound) or np.any(points<-bound))

        return is_in_bounds
    
    def set_motor_arm_angle(self, angles):
        self.motor_arm_angle = np.array([angles[0], 90 + angles[1], 180 + angles[2], 270 + angles[3]])
        self.opt_change_design_par = True

    def get_motor_arm_angle(self):
        return self.motor_arm_angle
    
    def gradient_direction(self,x, y):
        theta = np.arctan2(y, x)
        tangent_theta = theta + np.pi / 2
        
        dx = np.cos(tangent_theta)
        dy = np.sin(tangent_theta)
    
        return [dx, dy, 0]

    def is_inside_torus(self,x, y, z, R, r):
        d = np.sqrt(x**2 + y**2)
        return (d - R)**2 + z**2 < r**2
    
    def random_point_on_circle(self, r):
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        point = np.array([x, y, 0])
        return point