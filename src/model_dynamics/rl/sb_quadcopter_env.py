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

    def __init__(self, arm_length = np.array([0.17,0.17,0.17,0.17]), REWARD = np.array([0.0, 0.5, 0.0]), env_id = 0, n_steps_train = 512 * 10):
        super(QuadcopterEnv, self).__init__()

        utils.EzPickle.__init__(self)
        self.action_space = Box(low=-1, high=1, shape=(4,))

        self.Max_propeller_speed = 4000
        self.Min_propeller_speed = 0
        self.n_steps_train = n_steps_train
        self.traj_length = n_steps_train
        self.MAX_POS = 10
        self.MAX_OMEGA = 5
        self.MAX_VEL = 10
        self.max_init_vel = 0
        self.max_init_omega = 0
        self.pos_threshold = 0.25
        self.freq = 4
        self.env_id = env_id
        self.next_n_obs = 10
        self.observation_space = Box(low=-1, high=1, shape=(92,))
        self.steps = 0
        self.init_num = 0
        self.model = mw.multirotor0()
        self.TARGET_POS = np.array([0.0, 0.0, 0.0])  # for example
        self.QUAT_DES = np.array([1.0, 0.0, 0.0, 0.0])  # for example
        self.R_DES = np.array([1.10, 0.0 , 0.0 , 0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 1.0])
        self.INIT_POS = np.array([0.0, 0.0, 0.0])  # for example
        self.INIT_VEL = np.array([0.0, 0.0, 0.0])  # for example
        self.INIT_OMEGA = np.array([0, 0.0, 0.0])  # for example
        self.traj_radius = 0.2
        #reward function
        self.DISTANCE_REWARD = REWARD[0]
        self.VELOCITY_REWARD = REWARD[1]
        self.ANGULAR_VELOCITY_PENALTY = -REWARD[2]

        #Design parameters
        self.arm_length = np.array([0.17,0.17,0.17,0.17])
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

        self.prev_action = np.array([0.0, 0.0, 0.0, 0.0])
        self.opt_change_len = False
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
        obs_next_n_desired_pos = []
        pos_world = self.model.get_pos_world()
        # obs_position = pos_world - self.desired_path[self.steps]
        obs_position = pos_world       
        obs_position = np.clip(obs_position, -self.MAX_POS, self.MAX_POS)
        obs_position_norm = self.convert_range(obs_position, -self.MAX_POS, self.MAX_POS, -1, 1) 
        obs_position_des = self.desired_path[self.steps]
        obs_position_des_norm = self.convert_range(obs_position_des, -self.MAX_POS, self.MAX_POS, -1, 1)
        if self.steps < self.traj_length - self.next_n_obs:
            obs_next_n_desired_pos = np.array([p - pos_world for p in self.desired_path[self.steps+1:self.steps+1+self.next_n_obs]]).flatten()
            obs_next_n_desired_pos_norm = self.convert_range(obs_next_n_desired_pos, -self.MAX_POS, self.MAX_POS, -1, 1)
        else:
            obs_next_n_desired_pos = np.array([self.desired_path[self.traj_length-self.next_n_obs] for _ in range(self.next_n_obs)]).flatten()
            obs_next_n_desired_pos_norm = self.convert_range(obs_next_n_desired_pos, -self.MAX_POS, self.MAX_POS, -1, 1)
        # if self.steps < self.traj_length - self.next_n_obs:
        #     obs_next_n_desired_pos = np.array(self.desired_path[self.steps+1:self.steps+1+self.next_n_obs]).flatten()
        # else:
        #     obs_next_n_desired_pos = np.array([self.desired_path[self.traj_length-self.next_n_obs] for _ in range(self.next_n_obs)]).flatten()
        
        #get rotation matrix
        R = self.model.get_RotationMatrix_world()
        obs_rotation_matrix = np.clip(R, -1, 1)
        obs_rotation_matrix = obs_rotation_matrix.flatten()

        #get velocity
        obs_next_n_desired_vel = []
        vel_world = self.model.get_velocity_world()
        vel_world = np.clip(vel_world, -self.MAX_VEL, self.MAX_VEL)
        # obs_velocity = vel_world - self.desired_velocities[self.steps]
        obs_velocity = vel_world
        obs_velocity_norm = self.convert_range(obs_velocity, -self.MAX_VEL, self.MAX_VEL, -1, 1)
        obs_velocity_des = np.clip(self.desired_velocities[self.steps], -self.MAX_VEL, self.MAX_VEL)
        obs_velocity_des_norm = self.convert_range(obs_velocity_des, -self.MAX_VEL, self.MAX_VEL, -1, 1)    
        # if self.steps < self.traj_length - self.next_n_obs:
        #     obs_next_n_desired_vel = np.array([v - vel_world for v in self.desired_velocities[self.steps+1 : self.steps + self.next_n_obs + 1]]).flatten()
        # else:
        #     obs_next_n_desired_vel = np.zeros(3 * (self.next_n_obs))

        if self.steps < self.traj_length - self.next_n_obs:
            obs_next_n_desired_vel = np.array(self.desired_velocities[self.steps+1 : self.steps + self.next_n_obs + 1]).flatten()
            obs_next_n_desired_vel_norm = self.convert_range(obs_next_n_desired_vel, -self.MAX_VEL, self.MAX_VEL, -1, 1)
        else:
            obs_next_n_desired_vel = np.zeros(3 * (self.next_n_obs))
            obs_next_n_desired_vel_norm = np.zeros(3 * (self.next_n_obs))
        #get angular velocity
        omega_world = self.model.get_omega_world()
        omega_world = np.clip(omega_world, -self.MAX_OMEGA, self.MAX_OMEGA)
        obs_omega_norm = self.convert_range(omega_world, -self.MAX_OMEGA, self.MAX_OMEGA, -1, 1)

        #clip arm length
        obs_arm_length = np.clip(self.arm_length, 0.1, 1.0)
        obs_arm_length = self.convert_range(obs_arm_length, 0.1, 1.0, -1, 1)

        #get propeller speed
        propeller_speed = self.model.get_motor_rpm()
        propeller_speed = np.clip(propeller_speed, self.Min_propeller_speed, self.Max_propeller_speed)
        obs_propeller_speed_norm = self.convert_range(propeller_speed, self.Min_propeller_speed, self.Max_propeller_speed, -1, 1)

        #reward function
        reward_desired_pos = abs(np.linalg.norm(pos_world - self.desired_path[self.steps]))
        reward_desired_vel = abs(np.linalg.norm(vel_world - self.desired_velocities[self.steps]))
        
        reward_desired_pos_inter = 1 / (reward_desired_pos + 1e-6)
        reward_desired_vel_inter = 1 / (reward_desired_vel + 1e-6)

        reward_desired_pos_inter = np.clip(reward_desired_pos_inter, 0, 1/0.01)
        reward_desired_vel_inter = np.clip(reward_desired_vel_inter, 0, 1/0.01)
        
        reward_desired_pos_norm = self.convert_range(reward_desired_pos_inter, 0, 1/0.01, 0, 1)
        reward_desired_vel_norm = self.convert_range(reward_desired_vel_inter, 0, 1/0.01, 0, 1)

        reward_ang_vel = np.linalg.norm(omega_world)
        reward_ang_vel_norm = self.convert_range(reward_ang_vel, -self.MAX_OMEGA, self.MAX_OMEGA, 0, 1)
        reward_ang_vel_norm = np.clip(reward_ang_vel_norm, 0, 1)

        reward = self.DISTANCE_REWARD * reward_desired_pos_norm + self.VELOCITY_REWARD * reward_desired_vel_norm + self.ANGULAR_VELOCITY_PENALTY * reward_ang_vel_norm
        
        # if self.env_id == 0:
        #     print("reward: ", reward, "vel_world: ", vel_world - self.desired_velocities[self.steps], "pos_world: ", pos_world - self.desired_path[self.steps])

        if(self.steps >= self.traj_length - 1):
            done = True
            self.steps = 0
            print("steps exceeded")

        if not self.is_within_radius(desired_point = np.array(self.desired_path[self.steps]), current_point = np.array(pos_world), radius = self.traj_radius):
            done = True
            reward = -1.0

        if (np.any(np.isnan(pos_world)) or np.any(np.isinf(pos_world)) or
            np.any(np.isnan(vel_world)) or np.any(np.isinf(vel_world)) or
            np.any(np.isnan(omega_world)) or np.any(np.isinf(omega_world)) or
            np.any(np.isnan(R)) or np.any(np.isinf(R))):
            print("nan or inf detected")
            observation = np.ones(92)
            self.prev_action = action
            observation = observation.astype(np.float32)
            reward = -1.0
            reward = float(reward)
            early_Stop = False
            done  = True
            return observation, reward, done, early_Stop, {"pos_world": pos_world, "target_pos": self.TARGET_POS, "init_pos": self.INIT_POS, "rot_matrix_world":R.flatten(), "vel_world": vel_world, "omega_world": omega_world, "action": action, "reward": reward, "propeller_speed": propeller_speed, "action_motor_rpm": w, "arm_length": self.arm_length, "distance": reward_desired_pos, "target_reward_norm": reward_desired_pos_norm, "action_penalty": 0.0, "omega_world_penalty": 0.0}
        
        observation = np.concatenate([obs_position_norm, 
                                      obs_position_des_norm, 
                                      obs_velocity_norm, 
                                      obs_velocity_des_norm,
                                      obs_omega_norm, 
                                      obs_rotation_matrix, 
                                      obs_propeller_speed_norm, 
                                      obs_next_n_desired_pos_norm, 
                                      obs_next_n_desired_vel_norm, 
                                      obs_arm_length])     
        self.prev_action = action

        if self.is_within_radius(desired_point = np.array(self.desired_path[self.steps]), current_point = np.array(pos_world), radius = self.traj_radius/2):
            self.prev_propeller_speed = propeller_speed

        observation = observation.astype(np.float32)
        reward = float(reward)
        early_Stop = False
        return observation, reward, done, early_Stop, {"pos_world": pos_world, "target_pos": self.desired_path[self.steps], "init_pos": self.INIT_POS, "rot_matrix_world":R.flatten(), "vel_world": vel_world, "omega_world": omega_world, "action": action, "reward": reward, "propeller_speed": propeller_speed, "action_motor_rpm": w, "arm_length": self.arm_length, "desired_velocity" : self.desired_velocities[self.steps]}
    
    def reset(self, seed=None):
        
        if self.steps == 0:
            self.trajectory = trajectory()  # Trajectory object
            t_values = np.linspace(0, 1, self.traj_length)
            self.desired_path = [self.trajectory.get_trajectory_point(t) for t in t_values]
            self.desired_velocities = [self.trajectory.get_trajectory_velocity(t) for t in t_values]
            while not self.is_traj_in_bounds(self.desired_path, self.MAX_POS):
                self.trajectory = trajectory()
                t_values = np.linspace(0, 1, self.traj_length)
                self.desired_path = [self.trajectory.get_trajectory_point(t) for t in t_values]
                self.desired_velocities = [self.trajectory.get_trajectory_velocity(t) for t in t_values]
        
        print("reset", self.steps)

        if self.steps == 0:
            self.INIT_VEL = self.desired_velocities[self.steps]
            self.model.set_w_0([0.0, 0.0, 0.0, 0.0])
            if not self.opt_change_len:
                self.arm_length = np.array([random.uniform(0.1, 1.0), random.uniform(0.1, 1.0), random.uniform(0.1, 1.0), random.uniform(0.1, 1.0)])
        else:
            self.INIT_VEL = np.array([0.0, 0.0, 0.0])
            self.model.set_w_0(self.prev_propeller_speed)

        self.INIT_POS = self.desired_path[self.steps]
        self.INIT_OMEGA = np.array([random.uniform(-self.max_init_omega, self.max_init_omega), random.uniform(-self.max_init_omega, self.max_init_omega),random.uniform(-self.max_init_omega, self.max_init_omega)], dtype=np.float64)
        
        # #set design parameters and the rpm
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

        self.model.initialize()
        self.model.step2() #sets the propeller speeds
        self.model.step0()
        
        propeller_speed = self.model.get_motor_rpm()
        propeller_speed = np.clip(propeller_speed, self.Min_propeller_speed, self.Max_propeller_speed)
        obs_propeller_speed_norm = self.convert_range(propeller_speed, self.Min_propeller_speed, self.Max_propeller_speed, -1, 1)
        
        obs_next_n_desired_pos = []
        obs_next_n_desired_vel = []
        pos_world = self.model.get_pos_world()
        obs_position = pos_world
        obs_position = np.clip(obs_position, -self.MAX_POS, self.MAX_POS)
        obs_position_norm = self.convert_range(obs_position, -self.MAX_POS, self.MAX_POS, -1, 1)
        obs_position_des = self.desired_path[self.steps]
        obs_position_des_norm = self.convert_range(obs_position_des, -self.MAX_POS, self.MAX_POS, -1, 1)
        # obs_position = pos_world - self.desired_path[self.steps]
        if self.steps < self.traj_length - self.next_n_obs:
            obs_next_n_desired_pos = np.array([p - pos_world for p in self.desired_path[self.steps+1:self.steps+1+self.next_n_obs]]).flatten()
            obs_next_n_desired_pos_norm = self.convert_range(obs_next_n_desired_pos, -self.MAX_POS, self.MAX_POS, -1, 1)
        else:
            obs_next_n_desired_pos = np.array([self.desired_path[self.traj_length-self.next_n_obs] for _ in range(self.next_n_obs)]).flatten()
            obs_next_n_desired_pos_norm = self.convert_range(obs_next_n_desired_pos, -self.MAX_POS, self.MAX_POS, -1, 1)
        
        # if self.steps < self.traj_length - self.next_n_obs:
        #     obs_next_n_desired_pos = np.array(self.desired_path[self.steps+1:self.steps+1+self.next_n_obs]).flatten()
        # else:
        #     obs_next_n_desired_pos = np.array([self.desired_path[self.traj_length-self.next_n_obs] for _ in range(self.next_n_obs)]).flatten()
        
        vel_world = self.model.get_velocity_world()
        vel_world = np.clip(vel_world, -self.MAX_VEL, self.MAX_VEL)
        obs_velocity = vel_world
        obs_velocity_norm = self.convert_range(obs_velocity, -self.MAX_VEL, self.MAX_VEL, -1, 1)
        obs_velocity_des = np.clip(self.desired_velocities[self.steps], -self.MAX_VEL, self.MAX_VEL)
        obs_velocity_des_norm = self.convert_range(obs_velocity_des, -self.MAX_VEL, self.MAX_VEL, -1, 1)

        # obs_velocity = vel_world - self.desired_velocities[self.steps]
        # if self.steps < self.traj_length - self.next_n_obs:
        #     obs_next_n_desired_vel = np.array([v - vel_world for v in self.desired_velocities[self.steps+1 : self.steps + self.next_n_obs + 1]]).flatten()
        # else:
        #     obs_next_n_desired_vel = np.zeros(3 * (self.next_n_obs))

        if self.steps < self.traj_length - self.next_n_obs:
            obs_next_n_desired_vel = np.array(self.desired_velocities[self.steps+1 : self.steps + self.next_n_obs + 1]).flatten()
            obs_next_n_desired_vel_norm = self.convert_range(obs_next_n_desired_vel, -self.MAX_VEL, self.MAX_VEL, -1, 1)
        else:
            obs_next_n_desired_vel = np.zeros(3 * (self.next_n_obs))
            obs_next_n_desired_vel_norm = np.zeros(3 * (self.next_n_obs))
        
        #get rotation matrix
        R = self.model.get_RotationMatrix_world()
        R = np.clip(R, -1, 1)
        obs_rotation_matrix = R.flatten()

        #get angular velocity
        omega_world = self.model.get_omega_world()
        if(np.any(omega_world >= self.MAX_OMEGA) or np.any(omega_world <= -self.MAX_OMEGA)):
            reward = -1.0
        omega_world = np.clip(omega_world, -self.MAX_OMEGA, self.MAX_OMEGA)
        obs_omega_norm = self.convert_range(omega_world, -self.MAX_OMEGA, self.MAX_OMEGA, -1, 1)
    
        #clip arm length
        obs_arm_length = np.clip(self.arm_length, 0.1, 1.0)
        obs_arm_length = self.convert_range(obs_arm_length, 0.1, 1.0, -1, 1)
        observation = np.concatenate([obs_position_norm, 
                                      obs_position_des_norm, 
                                      obs_velocity_norm, 
                                      obs_velocity_des_norm,
                                      obs_omega_norm, 
                                      obs_rotation_matrix, 
                                      obs_propeller_speed_norm, 
                                      obs_next_n_desired_pos_norm, 
                                      obs_next_n_desired_vel_norm, 
                                      obs_arm_length])  
        
        observation = observation.astype(np.float32)
        info = {}

        if (np.any(np.isnan(pos_world)) or np.any(np.isinf(pos_world)) or
            np.any(np.isnan(vel_world)) or np.any(np.isinf(vel_world)) or
            np.any(np.isnan(omega_world)) or np.any(np.isinf(omega_world)) or
            np.any(np.isnan(R)) or np.any(np.isinf(R))):
            print("nan or inf detected in reset")
            observation = np.ones(92)
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
        self.opt_change_len = True
  
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
    
    def set_traj_radius(self, radius):
        self.traj_radius = radius

    def get_traj_radius(self):
        return self.traj_radius
    
    def is_traj_in_bounds(self, points, bound):

        points = np.array(points)
        is_in_bounds = not (np.any(points>bound) or np.any(points<-bound))

        return is_in_bounds
    
    def set_propeller_angles(self, angles):
        self.motor_arm_angle = angles
        self.opt_change_len = True

    def get_arm_length(self):
        return self.motor_arm_angle