import time
from typing import Any, Dict
import numpy as np
import gym
import pandas as pd
import os
import shutil
import xml.etree.ElementTree as ET
from stable_baselines3 import PPO
from quadcopter_racing_v1 import QuadcopterEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from scipy.io import loadmat, savemat

import random
import torch
import math 
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gaussMix_design_opt import DesignDistribution_log as DesignDistribution
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from scipy.io import loadmat, savemat

import random
import torch
import math
from sklearn.mixture import GaussianMixture


DESIGN_PARAMS = [
    {"name": "x1", "def_value": 0.1},
    {"name": "x2", "def_value": 0.1}
]

def main():
    #training parameters
    use_sde = False
    hidden_sizes_train = 64
    REWARD = np.array([1.0, 0.0])
    learning_rate_train = 0.0001
    n_epochs_train = 10
    LOAD_OLD_MODEL = False
    n_steps_train = 512 * 10
    n_envs_train = 8
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 1000
    batch_size_train = 128
    global_iteration = 0
    TRAIN = True
    CALL_BACK_FUNC = f"constant_design"

    original_xml_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/quadcopter.xml"
    destination_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/"

    for i in range(n_envs_train):
        new_file_name = f"quadcopter_{i}.xml"
        new_file_path = os.path.join(destination_folder, new_file_name)
        shutil.copy2(original_xml_path, new_file_path)
        # print(f"Copied to: {new_file_path}")


    while True:

        learning_rate_train = learning_rate_train

        onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                               net_arch=dict(vf=[hidden_sizes_train, hidden_sizes_train],
                                             pi=[hidden_sizes_train, hidden_sizes_train]))

        global_iteration += 1


        env_configs = [{'env_id': i, 'ctrl_cost_weight': 0.5} for i in range(n_envs_train)]

        assert len(env_configs) == n_envs_train


        env_fns = [lambda config=config: QuadcopterEnv(**config) for config in env_configs]

        vec_env = SubprocVecEnv(env_fns, start_method='fork')

        n_envs_eval = 1
        env_configs_eval = [{'env_id': i, 'ctrl_cost_weight': 0.5, 'render_mode': 'human'} for i in range(n_envs_eval)]

        assert len(env_configs_eval) == n_envs_eval

        env_fns_eval = [lambda config=config: QuadcopterEnv(**config) for config in env_configs_eval]

        vec_env_eval = DummyVecEnv(env_fns_eval)


        model_name = f"quadcopter_constant_design_racing_circular_3"
        log_dir = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/quadcopter_tensorboard/TB_{model_name}"

        if LOAD_OLD_MODEL is True:
            new_model = []
            old_model = PPO.load(f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/trained_model/quadcopter_constant_design_racing_circular_2.zip", env = vec_env)

            new_model = PPO("MlpPolicy", env=vec_env, n_steps=n_steps_train,
                            batch_size=batch_size_train, n_epochs=n_epochs_train,
                            use_sde=use_sde, ent_coef=entropy_coeff_train,
                            learning_rate=learning_rate_train, policy_kwargs=onpolicy_kwargs,
                            device='cpu', verbose=1, tensorboard_log=log_dir)

            new_model_eval = PPO("MlpPolicy", env=vec_env_eval, n_steps=n_steps_train,
                            batch_size=batch_size_train, n_epochs=n_epochs_train,
                            use_sde=use_sde, ent_coef=entropy_coeff_train,
                            learning_rate=learning_rate_train, policy_kwargs=onpolicy_kwargs,
                            device='cpu', verbose=1, tensorboard_log=log_dir)


            new_model.set_parameters(old_model.get_parameters())
            new_model_eval.set_parameters(old_model.get_parameters())

        else:
            new_model = PPO("MlpPolicy", env=vec_env, n_steps=n_steps_train, batch_size=batch_size_train,
                n_epochs=n_epochs_train, use_sde=use_sde, ent_coef=entropy_coeff_train,
                learning_rate=learning_rate_train,
                policy_kwargs=onpolicy_kwargs, device='cpu', verbose=1, tensorboard_log=log_dir)
            print("New model created")

        print("Model training...")
        if CALL_BACK_FUNC is f"constant_design":
            param_changer = constant_design(model_name = model_name, model = new_model, n_steps_train = n_steps_train, n_envs_train = n_envs_train, verbose=1)
        elif CALL_BACK_FUNC is f"random_design":
            param_changer = random_design(model_name = model_name, model = new_model, n_steps_train = n_steps_train, n_envs_train = n_envs_train, verbose=1)
        elif CALL_BACK_FUNC is f"Hebo_callback":
            param_changer = Hebo_callback(model_name=model_name, model=new_model, n_steps_train=n_steps_train, n_envs_train=n_envs_train, verbose=1)
        elif CALL_BACK_FUNC is f"Hebo_Gauss_callback":
            param_changer = Hebo_Gauss_callback(model_name = model_name, model = new_model, n_steps_train = n_steps_train, n_envs_train = n_envs_train, verbose=1)
        elif CALL_BACK_FUNC is f"Schaff_callback":
            param_changer = Schaff_callback(model_name=model_name, model=new_model, n_steps_train = n_steps_train, n_envs_train=n_envs_train, num_distributions=1, verbose=1)
        elif CALL_BACK_FUNC is f"Schaff_callback_GMM":
            param_changer = Schaff_callback_GMM(model_name=model_name, model=new_model, n_steps_train=n_steps_train, n_envs_train=n_envs_train, num_distributions=64, verbose=1)
        elif CALL_BACK_FUNC is f"evaluate_design":
            param_changer = evaluate_design(model_name = model_name, model = new_model_eval, n_steps_train = n_steps_train, n_envs_train = n_envs_eval, verbose=1)
        else:
            print("No callback function specified")
            break


        if TRAIN is True:
            new_model.learn(total_timesteps = total_timesteps_train ,progress_bar=True, callback=param_changer)
            print("Model trained, saving...")
            new_model.save(f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/trained_model/{model_name}")
            print("Model saved")
            LOAD_OLD_MODEL = True
            vec_env.close()
        else:
            new_model_eval.learn(total_timesteps = total_timesteps_train ,progress_bar=True, callback=param_changer)
            print("Model trained, saving...")
            LOAD_OLD_MODEL = True
            vec_env_eval.close()

        break


class constant_design(BaseCallback):
    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 10, n_envs_train=8, verbose=0):

        super(constant_design, self).__init__(verbose)
        self.model = model
        self.n_envs_train = n_envs_train
        self.n_steps_train = n_steps_train
        self.episode_rewards = {}
        self.rewards_iteration = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards = [0 for _ in range(self.n_envs_train)]
        self.episode_length = {}
        self.mat_design_params = []
        self.mat_reward = []
        self.mat_iteration = []
        self.average_reward = []
        self.average_episode_length = []
        self.model_name = model_name
        self.mat_file_name = model_name
        self.design_iteration = [0 for _ in range(self.n_envs_train)]
        self.my_custom_condition = True  # Initialize your condition
        self.model.evaluate_current_policy = False
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/"

        self.design_params = np.ones(8)

    def _on_rollout_start(self) -> bool:

        # reset the environments
        for i in range(self.n_envs_train):
            self.arm_1 = 0.1
            self.arm_2 = 0.1
            self.arm_3 = 0.1
            self.arm_4 = 0.1
            self.thruster_1 = 0.05
            self.thruster_2 = 0.05
            self.thruster_3 = 0.05
            self.thruster_4 = 0.05

            self.design_params = np.array([self.arm_1, self.arm_2, self.arm_3, self.arm_4, self.thruster_1, self.thruster_2, self.thruster_3, self.thruster_4])

            self.modify_xml_quadcopter_full_geometry(f"{self.mujoco_file_folder}quadcopter_{i}.xml", self.design_params)
            self.training_env.env_method('__init__', i, indices=[i])
            self.training_env.env_method("set_design_params", self.design_params, indices=[i])
            self.training_env.env_method('reset', indices=[i])
        return True

    def _on_step(self) -> bool:

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done or self.episode_length[i] >= self.n_steps_train:
                    # current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
                    # target_pos_tcp = self.training_env.env_method('get_target_pos_tcp', indices=[i])[0]
                    # dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                    self.average_episode_length.append(self.episode_length[i])
                    self.average_reward.append(self.episode_rewards[i])
                    self.design_iteration[i] += 1
                    # Reset episode reward accumulator
                    self.episode_rewards[i] = 0
                    self.episode_length[i] = 0
        return True

    def _on_rollout_end(self) -> bool:

        self.model.save(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/trained_model/{self.model_name}")


        for i in range(self.n_envs_train):
            self.mat_reward.append(self.episode_rewards[i])
            self.mat_iteration.append(self.episode_length[i])

        self.logger.record("mean episode length", np.sum(self.average_episode_length) / np.sum(self.design_iteration))
        self.logger.record("mean reward", np.sum(self.average_reward) / np.sum(self.design_iteration))

        output_data = {
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_iteration),
        }

        print("saving matlab data...")
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        self.average_episode_length = []
        self.average_reward = []
        self.design_iteration = [0 for _ in range(self.n_envs_train)]

        return True

    def modify_xml_quadcopter_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_paramss: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        original_lengths = {
            'arm': 0.05,
            'thruster': 0.05
        }


        [arm0, arm1, arm2, arm3, thruster0, thruster1, thruster2, thruster3] = design_params


        arms = root.findall(".//geom[@name='arm0']")
        for arm in arms:
            new_size = [str(arm0)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm1']")
        for arm in arms:
            new_size = [str(arm1)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm2']")
        for arm in arms:
            new_size = [str(arm2)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm3']")
        for arm in arms:
            new_size = [str(arm3)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))

        thrusters = root.findall(".//geom[@name='thruster0']")
        for thruster in thrusters:
            new_size = [str(thruster0)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))
        thrusters = root.findall(".//geom[@name='thruster1']")
        for thruster in thrusters:
            new_size = [str(thruster1)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster2']")
        for thruster in thrusters:
            new_size = [str(thruster2)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster3']")
        for thruster in thrusters:
            new_size = [str(thruster3)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))


        motors = root.findall(".//motor[@site='motor']")
        for motor in motors:
            new_gear = [str(0)] + [str(0)] + [str(arm0 * thruster0 / (original_lengths['arm'] * original_lengths['thruster']))] + [str(arm0 * arm0 * thruster0 / (original_lengths['arm'] * original_lengths['arm'] * original_lengths['thruster']))] + [str(arm0 * arm0 * thruster0 / (original_lengths['arm'] * original_lengths['arm'] * original_lengths['thruster']))] + [str(arm0 * arm0 * thruster0 / (original_lengths['arm'] * original_lengths['arm'] * original_lengths['thruster']))]
            motor.set('gear', ' '.join(new_gear))
        tree.write(file_path)


class random_design(BaseCallback):
    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 10, n_envs_train=8, verbose=0):

        super(random_design, self).__init__(verbose)
        self.model = model
        self.n_envs_train = n_envs_train
        self.n_steps_train = n_steps_train
        self.episode_rewards = {}
        self.rewards_iteration = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards = [0 for _ in range(self.n_envs_train)]
        self.episode_length = {}
        self.mat_design_params = []
        self.mat_reward = []
        self.mat_iteration = []
        self.average_reward = []
        self.average_episode_length = []
        self.model_name = model_name
        self.mat_file_name = model_name
        self.design_iteration = [0 for _ in range(self.n_envs_train)]
        self.my_custom_condition = True  # Initialize your condition
        self.model.evaluate_current_policy = False
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/"

        self.design_params = np.ones(8)
        self.arm_length_range = [0.1, 1.0]
        self.thruster_radius_range = [0.1, 1.0]

    def _on_rollout_start(self) -> bool:

        # reset the environments
        for i in range(self.n_envs_train):
            self.arm_1 = random.uniform(self.arm_length_range[0], self.arm_length_range[1])
            self.arm_2 = self.arm_1 #random.uniform(self.arm_length_range[0], self.arm_length_range[1])
            self.arm_3 = self.arm_1 #random.uniform(self.arm_length_range[0], self.arm_length_range[1])
            self.arm_4 = self.arm_1 #random.uniform(self.arm_length_range[0], self.arm_length_range[1])
            self.thruster_1 = random.uniform(self.thruster_radius_range[0], self.thruster_radius_range[1])
            self.thruster_2 = self.thruster_1 #random.uniform(self.thruster_radius_range[0], self.thruster_radius_range[1])
            self.thruster_3 = self.thruster_1 #random.uniform(self.thruster_radius_range[0], self.thruster_radius_range[1])
            self.thruster_4 = self.thruster_1 #random.uniform(self.thruster_radius_range[0], self.thruster_radius_range[1])
            self.design_params = np.array(
                [self.arm_1, self.arm_2, self.arm_3, self.arm_4, self.thruster_1, self.thruster_2, self.thruster_3,
                 self.thruster_4])

            self.modify_xml_quadcopter_full_geometry(f"{self.mujoco_file_folder}quadcopter_{i}.xml", self.design_params)
            self.training_env.env_method('__init__', i, indices=[i])
            self.training_env.env_method("set_design_params", self.design_params, indices=[i])
            self.training_env.env_method('reset', indices=[i])
        return True

    def _on_step(self) -> bool:

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done or self.episode_length[i] >= self.n_steps_train:
                    # current_limb_length = self.training_env.env_method('get_limb_length', indices=[i])[0]
                    # target_pos_tcp = self.training_env.env_method('get_target_pos_tcp', indices=[i])[0]
                    # dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                    self.average_episode_length.append(self.episode_length[i])
                    self.average_reward.append(self.episode_rewards[i])
                    self.design_iteration[i] += 1

                    self.episode_rewards[i] = 0
                    self.episode_length[i] = 0
        return True

    def _on_rollout_end(self) -> bool:

        self.model.save(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/trained_model/{self.model_name}")


        for i in range(self.n_envs_train):
            current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
            self.mat_design_params.append(current_design_params)
            self.mat_reward.append(self.episode_rewards[i])
            self.mat_iteration.append(self.episode_length[i])

        self.logger.record("mean episode length", np.sum(self.average_episode_length) / np.sum(self.design_iteration))
        self.logger.record("mean reward", np.sum(self.average_reward) / np.sum(self.design_iteration))

        output_data = {
            "design_params": np.array(self.mat_design_params),
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_iteration),
        }

        print("saving matlab data...")
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        self.average_episode_length = []
        self.average_reward = []
        self.design_iteration = [0 for _ in range(self.n_envs_train)]

        return True

    def modify_xml_quadcopter_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_paramss: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        original_lengths = {
            'arm': 0.05,
            'thruster': 0.05
        }

        [arm0, arm1, arm2, arm3, thruster0, thruster1, thruster2, thruster3] = design_params

        arms = root.findall(".//geom[@name='arm0']")
        for arm in arms:
            new_size = [str(arm0)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm1']")
        for arm in arms:
            new_size = [str(arm1)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm2']")
        for arm in arms:
            new_size = [str(arm2)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm3']")
        for arm in arms:
            new_size = [str(arm3)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))

        thrusters = root.findall(".//geom[@name='thruster0']")
        for thruster in thrusters:
            new_size = [str(thruster0)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))
        thrusters = root.findall(".//geom[@name='thruster1']")
        for thruster in thrusters:
            new_size = [str(thruster1)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster2']")
        for thruster in thrusters:
            new_size = [str(thruster2)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster3']")
        for thruster in thrusters:
            new_size = [str(thruster3)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        motors = root.findall(".//site[@name='motor0']")
        for motor in motors:
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor1']")
        for motor in motors:
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor2']")
        for motor in motors:
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor3']")
        for motor in motors:
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))

        motors = root.findall(".//motor[@site='motor0']")
        for motor in motors:
            new_gear = [str(0)] + [str(0)] + [
                str(arm0 * thruster0 / (original_lengths['arm'] * original_lengths['thruster']))] + [str(0)] + [
                           str(0)] + [
                           str(-arm0 * arm0 * thruster0 / (original_lengths['arm'] * original_lengths['thruster']))]
            motor.set('gear', ' '.join(new_gear))
        motors = root.findall(".//motor[@site='motor1']")
        for motor in motors:
            new_gear = [str(0)] + [str(0)] + [
                str(arm1 * thruster1 / (original_lengths['arm'] * original_lengths['thruster']))] + [str(0)] + [
                           str(0)] + [
                           str(arm1 * arm1 * thruster1 / (original_lengths['arm'] * original_lengths['thruster']))]
            motor.set('gear', ' '.join(new_gear))
        motors = root.findall(".//motor[@site='motor2']")
        for motor in motors:
            new_gear = [str(0)] + [str(0)] + [
                str(arm2 * thruster2 / (original_lengths['arm'] * original_lengths['thruster']))] + [str(0)] + [
                           str(0)] + [
                           str(-arm2 * arm2 * thruster2 / (original_lengths['arm'] * original_lengths['thruster']))]
            motor.set('gear', ' '.join(new_gear))
        motors = root.findall(".//motor[@site='motor3']")
        for motor in motors:
            new_gear = [str(0)] + [str(0)] + [
                str(arm3 * thruster3 / (original_lengths['arm'] * original_lengths['thruster']))] + [str(0)] + [
                           str(0)] + [
                           str(arm3 * arm3 * thruster3 / (original_lengths['arm'] * original_lengths['thruster']))]
            motor.set('gear', ' '.join(new_gear))
        tree.write(file_path)


class Schaff_callback(BaseCallback):
    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 2, n_envs_train=8, num_distributions=1,
                 verbose=0):

        super(Schaff_callback, self).__init__(verbose)
        self.num_distributions = num_distributions
        self.n_envs_train = n_envs_train
        self.model = model
        self.Schaffs_batch_size = 1
        self.episode_rewards = {}
        self.rewards_iteration = {}
        self.episode_length = {}
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        self.avg_design_iteration = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        self.episode_rewards = {}
        self.episode_length = {}
        self.logger_reward = []
        self.logger_episode_length = []
        self.model_name = model_name
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/"
        self.mat_file_name = model_name
        self.model.evaluate_current_policy = False
        self.save_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/random_design/{model_name}"
        # self.limb_length = np.ones(2) * 0.5

        self.distributions = []
        self.min_design_params = [0.1, 0.1]
        self.max_design_params = [1.0, 1.0]
        self.min_std = [0.0001, 0.0001]
        self.max_std = [0.5, 0.5]
        lr_std_schaff = 0.001
        lr_mean_schaff = 0.001
        lr_weight_schaff = 0.001
        self.mat_best_reward_policy = -1000
        self.mat_best_design = [0.5, 0.5]
        self.n_steps_train = n_steps_train
        self.steps_update_distribution = n_steps_train * n_envs_train * 500
        np.set_printoptions(precision=4)

        self.current_design_params = [[] for _ in range(self.n_envs_train)]
        self.mat_dist_mean = [[] for _ in range(self.n_envs_train)]
        self.mat_dist_std = [[] for _ in range(self.n_envs_train)]
        self.mat_dist_weight = [[] for _ in range(self.n_envs_train)]
        self.mat_design_params = [[] for _ in range(self.n_envs_train)]
        self.mat_reward = [[] for _ in range(self.n_envs_train)]
        self.mat_episode_length = [[] for _ in range(self.n_envs_train)]
        self.mat_iter = 0
        self.accumulated_rewards_chopping_metric = [[] for _ in range(self.n_envs_train)]

        self.start_chopping = False
        self.start_sampling_distributions = False
        self.iteration_matlab = 0
        self.init_pos = []

        self.checker = [False for _ in range(self.n_envs_train)]

        z = 2

        for _ in range(self.num_distributions):
            self.initial_mean = [0.5, 0.5]
            # self.initial_mean = np.array([min_val + (max_val - min_val) * np.random.rand()
            #                              for min_val, max_val in zip(self.min_limb_length, self.max_limb_length)])
            self.initial_std = [0.25, 0.25]  # Initialize std deviation as you prefer
            # self.initial_std = np.array([(max_Val - min_val) / (2 * z)
            #                             for min_val, max_val in
            #                             zip(self.min_limb_length, self.max_limb_length)])

            self.design_dist = DesignDistribution(self.initial_mean, self.initial_std,
                                                  min_parameters=self.min_design_params,
                                                  max_parameters=self.max_design_params, min_std=self.min_std,
                                                  max_std=self.max_std, lr_mean=lr_mean_schaff,
                                                  lr_std=lr_std_schaff, lr_weight=lr_weight_schaff)
            self.distributions.append(self.design_dist)

    def uniform_distribution_variance(self, a, b):
        """
        Calculate the variance and standard deviation of a uniform distribution
        with bounds a and b.
        """
        variance = ((b - a) ** 2) / 12
        std_dev = (variance ** 0.5)

        return std_dev

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        print("Training started")
        # set the environment id for each environment
        for i in range(self.n_envs_train):
            self.training_env.env_method('set_env_id', i, indices=[i])

        # print(f"Env IDs: {[self.training_env.env_method('get_env_id', indices=[i])[0] for i in range(self.n_envs_train)]}")

        return True

    def _on_rollout_start(self) -> bool:

        self.checker = [False for _ in range(self.n_envs_train)]

        # reset the environments
        for i in range(self.n_envs_train):
            self.training_env.env_method('reset', indices=[i])

        # set the sampled limb length for each batch
        for i in range(self.n_envs_train // self.Schaffs_batch_size):
            dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
            new_design_params = self.distributions[(0)].sample_design().detach().numpy()
            new_design_params = np.clip(new_design_params, self.min_design_params, self.max_design_params)
            new_design_params_update = np.array(
                [new_design_params[0], new_design_params[0], new_design_params[0], new_design_params[0],
                 new_design_params[1], new_design_params[1], new_design_params[1], new_design_params[1]])
            self.modify_xml_quadcopter_full_geometry(f"{self.mujoco_file_folder}quadcopter_{i}.xml", new_design_params_update)
            self.training_env.env_method('__init__', i, indices=[i])
            self.training_env.env_method("set_design_params", new_design_params_update, indices=[i])

        for i in range(self.n_envs_train // self.Schaffs_batch_size):
            for j in range(self.Schaffs_batch_size):
                self.training_env.env_method('reset', indices=[i * self.Schaffs_batch_size + j])
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.Schaffs_batch_size + j])[0]
                new_limb_length = \
                    self.training_env.env_method('get_design_params', indices=[i * self.Schaffs_batch_size + j])[0]
                # print(f"env id: {dist_env_id}, init pos: {init_pos}, limb length: {new_limb_length}, mean: {self.distributions[i//(self.n_envs_train//(self.Schaffs_batch_size*self.num_distributions))].get_mean()[0]}, std: {self.distributions[i//(self.n_envs_train//(self.Schaffs_batch_size*self.num_distributions))].get_std()[0]}")
        return True

    def _on_rollout_end(self) -> bool:

        # calculate the mean reward for each unique design (limb lengths and thicknesses)

        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        for i in range(self.n_envs_train // self.Schaffs_batch_size):
            # average batch reward
            sum_reward = 0
            total_episode_length = 0
            total_design_iteration = 0
            for j in range(self.Schaffs_batch_size):
                sum_reward += self.episode_rewards[i * self.Schaffs_batch_size + j] / self.design_iteration[
                    i * self.Schaffs_batch_size + j]
                total_episode_length += self.episode_length[i * self.Schaffs_batch_size + j] / self.design_iteration[
                    i * self.Schaffs_batch_size + j]
                total_design_iteration += self.design_iteration[i * self.Schaffs_batch_size + j]
            self.design_rewards_avg[i] = sum_reward / self.Schaffs_batch_size
            self.episode_length_avg[i] = total_episode_length / self.Schaffs_batch_size
            self.avg_design_iteration[i] = total_design_iteration / self.Schaffs_batch_size

        # update the design distribution based on the mean reward

        for i in range(self.n_envs_train // self.Schaffs_batch_size):
            self.current_design_params[i] = \
                self.training_env.env_method('get_design_params', indices=[i * self.Schaffs_batch_size])[0]
            self.current_design_params[i] = np.array(
                [self.current_design_params[i][0], self.current_design_params[i][4]])
            self.mat_dist_mean[i].append(self.distributions[(0)].get_mean())
            self.mat_dist_std[i].append(self.distributions[(0)].get_std())
            self.mat_dist_weight[i].append(self.distributions[(0)].get_weight())
            self.mat_design_params[i].append(self.current_design_params[i])
            self.mat_reward[i].append(self.design_rewards_avg[i])
            self.mat_episode_length[i].append(self.episode_length_avg[i])
            self.accumulated_rewards_chopping_metric[i].append(self.design_rewards_avg[i])
            print(
                f"env: {i * self.Schaffs_batch_size:<1.2f}, limb length: {self.current_design_params[i]}, mean reward: {self.design_rewards_avg[i]:<1.2f}, mean episode length: {self.episode_length_avg[i]:<1.2f}, design iteration: {self.avg_design_iteration[i]}, dist mean: {self.distributions[0].get_mean()}, dist std: {self.distributions[0].get_std()}, dist weight: {self.distributions[0].get_weight()}")

            self.logger_reward.append(self.design_rewards_avg[i])
            self.logger_episode_length.append(self.episode_length_avg[i])

            if self.design_rewards_avg[i] > self.mat_best_reward_policy:
                self.mat_best_reward_policy = self.design_rewards_avg[i]
                self.mat_best_design = self.current_design_params[i]
                self.model.save(
                    f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/bestDesign_{self.model_name}")

        if self.num_timesteps >= self.steps_update_distribution:
            self.distributions[(0)].update_distribution(
                [self.logger_reward],
                [self.current_design_params],
                self.n_envs_train)
        output_data = {
            "dist_mean": np.array(self.mat_dist_mean),
            "dist_std": np.array(self.mat_dist_std),
            "design_params": np.array(self.mat_design_params),
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_episode_length),
            "best_reward": self.mat_best_reward_policy,
            "best_design": np.array(self.mat_best_design)
        }

        self.logger.record("mean reward", np.mean(self.logger_reward))
        self.logger.record("mean episode length", np.mean(self.logger_episode_length))
        self.logger_reward = []
        self.logger_episode_length = []

        print("saving matlab data...")
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        print("saving current model...")
        self.model.save(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/{self.model_name}")

        print("model amd matlab data saved")

        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.episode_rewards = {}
        self.episode_length = {}

        return True

    def _on_step(self) -> bool:
        st = time.time()

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                if self.checker[i] == False:
                    self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                    self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done:
                    self.checker[i] = True

        return True


    def modify_xml_quadcopter_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_paramss: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        [arm0, arm1, arm2, arm3, thruster0, thruster1, thruster2, thruster3] = design_params

        arms = root.findall(".//geom[@name='arm0']")
        for arm in arms:
            new_size = [str(arm0)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm1']")
        for arm in arms:
            new_size = [str(arm1)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm2']")
        for arm in arms:
            new_size = [str(arm2)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm3']")
        for arm in arms:
            new_size = [str(arm3)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))

        thrusters = root.findall(".//geom[@name='thruster0']")
        for thruster in thrusters:
            new_size = [str(thruster0)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))
        thrusters = root.findall(".//geom[@name='thruster1']")
        for thruster in thrusters:
            new_size = [str(thruster1)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster2']")
        for thruster in thrusters:
            new_size = [str(thruster2)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster3']")
        for thruster in thrusters:
            new_size = [str(thruster3)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        motors = root.findall(".//site[@name='motor0']")
        for motor in motors:
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor1']")
        for motor in motors:
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor2']")
        for motor in motors:
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor3']")
        for motor in motors:
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        tree.write(file_path)


class Schaff_callback_GMM(BaseCallback):
    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 2, n_envs_train=50, num_distributions=50,
                 verbose=0):

        super(Schaff_callback_GMM, self).__init__(verbose)
        self.num_distributions = num_distributions
        self.n_envs_train = n_envs_train
        self.model = model
        self.Schaffs_batch_size = self.n_envs_train // self.num_distributions
        self.rewards_iteration = {}
        self.episode_length = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        self.avg_design_iteration = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        self.episode_rewards = {}
        self.episode_length = {}
        self.logger_reward = []
        self.logger_episode_length = []
        self.model_name = model_name
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/"
        self.mat_file_name = model_name
        self.model.evaluate_current_policy = False
        self.save_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/random_design/{model_name}"
        # self.limb_length = np.ones(2) * 0.5

        self.distributions = []
        self.min_design_params = [0.1, 0.1]
        self.max_design_params = [1.0, 1.0]
        self.min_std = [0.0001, 0.0001]
        self.max_std = [0.5, 0.5]
        lr_std_schaff = 0.001
        lr_mean_schaff = 0.001
        lr_weight_schaff = 0.001

        self.mat_best_reward_policy = -1000
        self.mat_best_design = [0.5, 0.5]

        self.n_steps_train = n_steps_train
        self.steps_update_distribution = n_steps_train * n_envs_train * 500
        self.steps_chop_distribution = n_steps_train * n_envs_train * 1000
        np.set_printoptions(precision=4)

        self.current_design_params = [[] for _ in range(self.num_distributions)]
        self.mat_dist_mean = [[] for _ in range(self.num_distributions)]
        self.mat_dist_std = [[] for _ in range(self.num_distributions)]
        self.mat_design_params = [[] for _ in range(self.num_distributions)]
        self.mat_reward = [[] for _ in range(self.num_distributions)]
        self.mat_episode_length = [[] for _ in range(self.num_distributions)]
        self.mat_iter = 0
        self.accumulated_rewards_chopping_metric = [[] for _ in range(self.num_distributions)]

        self.start_sampling_distributions = False
        self.iteration_matlab = 0
        self.accumulate = False
        self.counter_evaluations = 0
        self.init_pos = []

        self.checker = [False for _ in range(self.n_envs_train)]

        z = 2

        for _ in range(self.num_distributions):
            self.initial_mean = np.array([min_val + (max_val - min_val) * np.random.rand()
                                          for min_val, max_val in zip(self.min_design_params, self.max_design_params)])
            self.initial_std = [0.25, 0.25]  # Initialize std deviation as you prefer
            # self.initial_std = np.array([(max_Val - min_val) / (2 * z)
            #                             for min_val, max_val in
            #                             zip(self.min_limb_length, self.max_limb_length)])

            self.design_dist = DesignDistribution(self.initial_mean, self.min_design_params,
                                                  max_parameters=self.max_design_params, min_std=self.min_std,
                                                  max_std=self.max_std, lr_mean=lr_mean_schaff,
                                                  lr_std=lr_std_schaff, lr_weight=lr_weight_schaff)
            self.distributions.append(self.design_dist)

    def uniform_distribution_variance(self, a, b):
        """
        Calculate the variance and standard deviation of a uniform distribution
        with bounds a and b.
        """
        variance = ((b - a) ** 2) / 12
        std_dev = (variance ** 0.5)

        return std_dev

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        print("Training started")
        # set the environment id for each environment
        for i in range(self.n_envs_train):
            self.training_env.env_method('set_env_id', i, indices=[i])

        # print(f"Env IDs: {[self.training_env.env_method('get_env_id', indices=[i])[0] for i in range(self.n_envs_train)]}")

        return True

    def _on_rollout_start(self) -> bool:

        self.checker = [False for _ in range(self.n_envs_train)]

        # chop low performing distributions
        if self.num_timesteps > 0 and self.num_timesteps % self.steps_chop_distribution == 0 and self.num_distributions > 1:
            self.accumulate = True
            self.model.evaluate_current_policy = True

            self.counter_evaluations = 0
            self.accumulated_rewards_chopping_metric = [[] for _ in range(self.num_distributions)]

        # reset the environments
        for i in range(self.n_envs_train):
            self.training_env.env_method('reset', indices=[i])

        # set the sampled limb length for each batch
        for i in range(self.n_envs_train):
            dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
            new_design_params = self.distributions[
                (dist_env_id // (self.n_envs_train // self.num_distributions))].sample_design().detach().numpy()
            new_design_params = np.clip(new_design_params, self.min_design_params, self.max_design_params)
            new_design_params_update = np.array(
                [new_design_params[0], new_design_params[0], new_design_params[0], new_design_params[0],
                 new_design_params[1], new_design_params[1], new_design_params[1], new_design_params[1]])
            self.modify_xml_quadcopter_full_geometry(f"{self.mujoco_file_folder}quadcopter_{i}.xml", new_design_params_update)
            self.training_env.env_method('__init__', i, indices=[i])
            self.training_env.env_method("set_design_params", new_design_params_update, indices=[i])
            self.training_env.env_method('reset', indices=[i])
        return True

    def _on_rollout_end(self) -> bool:

        # calculate the mean reward for each unique design (limb lengths and thicknesses)

        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.Schaffs_batch_size)]
        for i in range(self.n_envs_train // self.Schaffs_batch_size):
            # average batch reward
            sum_reward = 0
            total_episode_length = 0
            total_design_iteration = 0
            for j in range(self.Schaffs_batch_size):
                sum_reward += self.episode_rewards[i * self.Schaffs_batch_size + j] / self.design_iteration[
                    i * self.Schaffs_batch_size + j]
                total_episode_length += self.episode_length[i * self.Schaffs_batch_size + j] / self.design_iteration[
                    i * self.Schaffs_batch_size + j]
                total_design_iteration += self.design_iteration[i * self.Schaffs_batch_size + j]

            self.design_rewards_avg[i] = sum_reward / self.Schaffs_batch_size
            self.episode_length_avg[i] = total_episode_length / self.Schaffs_batch_size
            self.avg_design_iteration[i] = total_design_iteration / self.Schaffs_batch_size

            for j in range(self.Schaffs_batch_size):
                self.current_design_params[i * self.Schaffs_batch_size + j] = \
                    self.training_env.env_method('get_design_params', indices=[i * self.Schaffs_batch_size + j])[0]
                self.current_design_params[i * self.Schaffs_batch_size + j] = np.array(
                    [self.current_design_params[i * self.Schaffs_batch_size + j][0],
                     self.current_design_params[i * self.Schaffs_batch_size + j][4]])
                print(
                    f"env: {i * self.Schaffs_batch_size:<1.2f}, design_params: {self.current_design_params[i * self.Schaffs_batch_size + j]}, mean reward: {self.episode_rewards[i * self.Schaffs_batch_size + j]:<1.2f}, mean episode length: {self.episode_length[i * self.Schaffs_batch_size + j]:<1.2f}, design iteration: {self.design_iteration[i * self.Schaffs_batch_size + j]}, dist mean: {self.distributions[i].get_mean()}, dist std: {self.distributions[i].get_std()}")
                self.mat_dist_mean[i * self.Schaffs_batch_size + j].append(
                    self.distributions[i // (self.n_envs_train // self.num_distributions)].get_mean())
                self.mat_dist_std[i * self.Schaffs_batch_size + j].append(
                    self.distributions[i // (self.n_envs_train // self.num_distributions)].get_std())
                self.mat_design_params[i * self.Schaffs_batch_size + j].append(
                    self.current_design_params[i * self.Schaffs_batch_size + j])
                self.mat_reward[i * self.Schaffs_batch_size + j].append(
                    self.episode_rewards[i * self.Schaffs_batch_size + j])
                self.mat_episode_length[i * self.Schaffs_batch_size + j].append(
                    self.episode_length[i * self.Schaffs_batch_size + j])

            self.logger_reward.append(self.design_rewards_avg[i])
            self.logger_episode_length.append(self.episode_length_avg[i])

            if self.design_rewards_avg[i] > self.mat_best_reward_policy:
                self.mat_best_reward_policy = self.design_rewards_avg[i]
                self.mat_best_design = self.current_design_params[i * self.Schaffs_batch_size]
                self.model.save(
                    f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/bestDesign_{self.model_name}")

            if self.num_timesteps >= self.steps_update_distribution and not self.accumulate:
                self.distributions[i].update_distribution(
                    [self.design_rewards_avg[i]],
                    [self.current_design_params[i]],
                    self.num_distributions)

        for i in range(self.num_distributions):
            if self.accumulate:
                self.accumulated_rewards_chopping_metric[i].append(self.design_rewards_avg[i])

        output_data = {
            "dist_mean": np.array(self.mat_dist_mean),
            "dist_std": np.array(self.mat_dist_std),
            "design_params": np.array(self.mat_design_params),
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_episode_length),
            "best_reward": self.mat_best_reward_policy,
            "best_design": np.array(self.mat_best_design)
        }

        self.logger.record("mean reward", np.mean(self.logger_reward))
        self.logger.record("mean episode length", np.mean(self.logger_episode_length))
        self.logger_reward = []
        self.logger_episode_length = []

        print("saving matlab data...")
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        print("saving current model...")
        self.model.save(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/{self.model_name}")

        print("model amd matlab data saved")

        if self.accumulate:
            self.counter_evaluations += 1

        print(self.counter_evaluations)

        # chop low performing distributions
        if self.counter_evaluations == 100 and self.num_distributions > 1:

            print("...Starting to chop distributions...")
            print("Updating design distribution...")

            print(f"Rewards: {self.accumulated_rewards_chopping_metric}")
            mean_rewards = [np.mean(self.accumulated_rewards_chopping_metric[i]) for i in
                            range(len(self.accumulated_rewards_chopping_metric))]

            sorted_indices = np.argsort(mean_rewards)[::-1]
            top_indices = sorted_indices[:len(sorted_indices) // 2]

            self.num_distributions = len(top_indices)
            self.Schaffs_batch_size = self.n_envs_train // self.num_distributions
            # self.n_envs_train = len(top_indices)
            self.distributions = [self.distributions[i] for i in top_indices]

            print(
                f"Top indices: {top_indices}, mean rewards over the phase: {mean_rewards}, sorted indices: {sorted_indices}")

            print(f"Kept {len(top_indices)} top-performing distributions.")
            print(
                f"New distribution means: {[self.distributions[i].get_mean() for i in range(len(self.distributions))]}")
            print(
                f"New distribution stds: {[self.distributions[i].get_std() for i in range(len(self.distributions))]}")
            print(
                f"New distribution env IDs: {[self.training_env.env_method('get_env_id', indices=[i])[0] for i in range(self.num_distributions)]}")
            print(
                f"New distribution mean rewards: {[np.mean(self.accumulated_rewards_chopping_metric[i]) for i in range(len(self.accumulated_rewards_chopping_metric))]}")

            self.accumulated_rewards_chopping_metric = [[] for _ in range(self.num_distributions)]
            self.accumulate = False
            self.counter_evaluations = 0
            self.model.evaluate_current_policy = False

            # self.mat_dist_mean = [self.mat_dist_mean[i] for i in top_indices]
            # self.mat_dist_std = [self.mat_dist_std[i] for i in top_indices]
            # self.mat_limb_length = [self.mat_limb_length[i] for i in top_indices]
            # self.mat_reward = [self.mat_reward[i] for i in top_indices]
            # self.mat_episode_length = [self.mat_episode_length[i] for i in top_indices]

            for i in range(self.n_envs_train // self.Schaffs_batch_size):
                for j in range(self.Schaffs_batch_size):
                    self.training_env.env_method('set_env_id', i,
                                                 indices=[i * self.Schaffs_batch_size])

        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.checker = [False for _ in range(self.n_envs_train)]
        self.episode_rewards = {}
        self.episode_length = {}

        return True

    def _on_step(self) -> bool:
        st = time.time()

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                if i < len(self.checker):
                    if self.checker[i] == False:
                        self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                        self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done:
                    if i < len(self.checker):
                        self.checker[i] = True

        return True

    def modify_xml_quadcopter_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_paramss: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        [arm0, arm1, arm2, arm3, thruster0, thruster1, thruster2, thruster3] = design_params

        arms = root.findall(".//geom[@name='arm0']")
        for arm in arms:
            new_size = [str(arm0)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm1']")
        for arm in arms:
            new_size = [str(arm1)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm2']")
        for arm in arms:
            new_size = [str(arm2)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm3']")
        for arm in arms:
            new_size = [str(arm3)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))

        thrusters = root.findall(".//geom[@name='thruster0']")
        for thruster in thrusters:
            new_size = [str(thruster0)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))
        thrusters = root.findall(".//geom[@name='thruster1']")
        for thruster in thrusters:
            new_size = [str(thruster1)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster2']")
        for thruster in thrusters:
            new_size = [str(thruster2)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster3']")
        for thruster in thrusters:
            new_size = [str(thruster3)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        motors = root.findall(".//site[@name='motor0']")
        for motor in motors:
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor1']")
        for motor in motors:
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor2']")
        for motor in motors:
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor3']")
        for motor in motors:
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        tree.write(file_path)


class Hebo_callback(BaseCallback):

    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 10, n_envs_train=100, verbose=0):

        super(Hebo_callback, self).__init__(verbose)

        self.batch_iterations = n_steps_train * n_envs_train
        self.steps_update_distribution = self.batch_iterations * 1  # Set to batch_iterations * 1 for clarity
        self.n_envs_train = n_envs_train
        self.model_name = model_name
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/"

        self.distributions = []
        self.mat_design_params = []
        self.mat_reward = []
        self.mat_iteration = []
        self.mat_time = []
        self.state = 'propose_design'
        self.model = model
        self.design_process = False  # Initialize to False
        self.mat_file_name = model_name
        self.save_recorded_data = n_steps_train * n_envs_train * 1
        self.reduce_batch_size = n_steps_train * n_envs_train * 1
        self.model.evaluate_current_policy = True
        self.batch_size_opt = 1
        self.n_steps_train = 10000
        # self.model.evaluate_current_policy = True

        # Initialize
        self.episode_rewards = {}
        self.episode_length = {}
        self.episode_times = {}
        self.convergence_steps = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
        self.best_design = []
        self.best_design_reward = []
        self.average_length = []
        self.average_reward = []
        self.average_time = []
        self.checker = [False for _ in range(self.n_envs_train)]

        np.set_printoptions(precision=5)

        space = DesignSpace().parse([
            {'name': 'x1', 'type': 'num', 'lb': 0.1, 'ub': 0.5},
            {'name': 'x2', 'type': 'num', 'lb': 0.1, 'ub': 1.0},
        ])

        self.opt = HEBO(space)
        print("Initialisation callback: ")

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        print("Training started")
        print(self.n_envs_train)
        # set the environment id for each environment
        for i in range(self.n_envs_train):
            self.training_env.env_method('set_env_id', i, indices=[i])

        return True

    def _on_rollout_start(self) -> bool:

        self.checker = [False for _ in range(self.n_envs_train)]
        self.average_reward = []
        self.average_length = []
        self.average_time = []

        self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.episode_rewards = {}
        self.episode_length = {}
        self.episode_times = {}
        self.convergence_steps = {}

        print("...Updating the new lengths...")
        start_time = time.time()

        self.rec = self.opt.suggest(n_suggestions=self.n_envs_train // self.batch_size_opt)

        for i in range(self.n_envs_train // self.batch_size_opt):
            for j in range(self.batch_size_opt):
                new_design_params = []
                curr_rec = self.rec.iloc[i]
                for design_param in DESIGN_PARAMS:
                    try:
                        new_design_params.append(curr_rec.at[design_param["name"]])
                    except:
                        new_design_params.append(design_param["def_value"])
                new_design_params = np.array(new_design_params)
                new_design_params = np.array(
                    [new_design_params[0], new_design_params[0], new_design_params[0], new_design_params[0],
                     new_design_params[1], new_design_params[1], new_design_params[1], new_design_params[1]])

                self.modify_xml_quadcopter_full_geometry(f"{self.mujoco_file_folder}quadcopter_{i}.xml",
                                                     new_design_params)
                self.training_env.env_method('__init__', i, indices=[i])
                self.training_env.env_method('set_design_params', new_design_params,
                                             indices=[i * self.batch_size_opt + j])
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                print(
                    f"env: {i * self.batch_size_opt + j:<1.2f}, real id:{dist_env_id}, design parameters: {new_design_params}")

        print(f"Design proposal took {time.time() - start_time:.2f} seconds")
        self.training_env.env_method('reset', indices=range(self.n_envs_train))
        return True

    def _on_step(self) -> bool:
        st = time.time()

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                if self.checker[i] == False:
                    self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                    self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done:
                    self.checker[i] = True

        return True

    def _on_rollout_end(self) -> bool:

        if self.num_timesteps >= self.steps_update_distribution - self.n_envs_train:

            print("...Starting design distribution update...")
            start_time = time.time()

            scores = []

            self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
            self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
            for i in range(self.n_envs_train // self.batch_size_opt):
                # average batch reward
                sum_reward = 0
                total_episode_length = 0
                for j in range(self.batch_size_opt):
                    sum_reward += self.episode_rewards[i * self.batch_size_opt + j] / self.design_iteration[
                        i * self.batch_size_opt + j]

                    total_episode_length += self.episode_length[i * self.batch_size_opt + j] / self.design_iteration[
                        i * self.batch_size_opt + j]

                self.design_rewards_avg[i] = sum_reward / self.batch_size_opt
                self.episode_length_avg[i] = total_episode_length / self.batch_size_opt

                self.average_length.append(self.episode_length_avg[i])
                self.average_reward.append(self.design_rewards_avg[i])

                score_array = np.array(self.design_rewards_avg[i]).reshape(-1, 1)
                scores.append(-score_array)

                current_design_params = \
                    self.training_env.env_method('get_design_params', indices=[i * self.batch_size_opt])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.batch_size_opt])[0]
                print(
                    f"Env ID: {dist_env_id}, mean reward: {self.design_rewards_avg[i]}, Mean episode length: {self.episode_length_avg[i]}, arm length: {current_design_params}")

                self.mat_design_params.append(current_design_params)
                self.mat_reward.append(self.design_rewards_avg[i])
                self.mat_iteration.append(self.episode_length_avg[i])

            self.logger.record("mean reward", np.sum(self.average_reward) / (self.n_envs_train // self.batch_size_opt))
            self.logger.record("mean episode length",
                               np.sum(self.average_length) / (self.n_envs_train // self.batch_size_opt))

            scores = np.array(scores)

            self.opt.observe(self.rec, scores)

            self.state = 'propose_design'

            best_idx = self.opt.y.argmin()
            best_design = self.opt.X.iloc[best_idx]
            best_design_reward = self.opt.y[best_idx]

            print(f"Best design: {best_design}, best reward: {best_design_reward}")
            self.best_design.append(best_design)
            self.best_design_reward.append(best_design_reward)

            print(f"Design distribution update took {time.time() - start_time:.2f} seconds")
        else:

            for i in range(self.n_envs_train):
                current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                print(
                    f"Env ID: {dist_env_id}, episode reward: {self.episode_rewards[i]}, mean reward: {self.episode_rewards[i] / self.design_iteration[i]}, design iter: {self.design_iteration[i]}, episode length: {self.episode_length[i]}, arm length: {current_design_params}")
                self.logger.record("mean reward", self.episode_rewards[i] / self.design_iteration[i])
                self.logger.record("mean episode length", self.episode_length[i])

                # Matlab logging
                self.mat_design_params.append(current_design_params)
                self.mat_reward.append(self.episode_rewards[i] / self.design_iteration[i])
                self.mat_iteration.append(self.episode_length[i] / self.design_iteration[i])

        if self.num_timesteps % self.save_recorded_data == 0:
            output_data = {
                "design_params": np.array(self.mat_design_params),
                "reward": np.array(self.mat_reward),
                "iteration": np.array(self.mat_iteration)
            }
            print("saving matlab data...")

            file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/quadcopter/{self.mat_file_name}.mat"
            savemat(file_path, output_data)
            print("saving current model...")
            self.model.save(
                f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/quadcopter/rl/trained_model/HEBO/{self.model_name}")
            print("Model saved")

        # increasing the batch size with each iteration
        # if self.num_timesteps % self.reduce_batch_size == 0:
        #     print("Increasing batch size...")
        #     self.batch_size_opt = 2 * self.batch_size_opt
        #     if self.batch_size_opt > self.n_envs_train:
        #         self.batch_size_opt = self.n_envs_train

        return True


    def modify_xml_quadcopter_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_paramss: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        [arm0, arm1, arm2, arm3, thruster0, thruster1, thruster2, thruster3] = design_params

        arms = root.findall(".//geom[@name='arm0']")
        for arm in arms:
            new_size = [str(arm0)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm1']")
        for arm in arms:
            new_size = [str(arm1)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm2']")
        for arm in arms:
            new_size = [str(arm2)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm3']")
        for arm in arms:
            new_size = [str(arm3)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))

        thrusters = root.findall(".//geom[@name='thruster0']")
        for thruster in thrusters:
            new_size = [str(thruster0)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))
        thrusters = root.findall(".//geom[@name='thruster1']")
        for thruster in thrusters:
            new_size = [str(thruster1)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster2']")
        for thruster in thrusters:
            new_size = [str(thruster2)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster3']")
        for thruster in thrusters:
            new_size = [str(thruster3)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        motors = root.findall(".//site[@name='motor0']")
        for motor in motors:
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor1']")
        for motor in motors:
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor2']")
        for motor in motors:
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor3']")
        for motor in motors:
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        tree.write(file_path)


class Hebo_Gauss_callback(BaseCallback):

    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 10, n_envs_train=8,
                 limb_length_limits=np.array([0.4, 2.0]), verbose=0):

        super(Hebo_Gauss_callback, self).__init__(verbose)

        self.batch_iterations = n_steps_train * n_envs_train
        self.steps_update_distribution = self.batch_iterations * 0  # Set to batch_iterations * 1 for clarity
        self.n_envs_train = n_envs_train
        self.model_name = model_name
        self.model = model
        self.distributions = []
        self.mat_design_params = []
        self.mat_best_design_gauss = []
        self.mat_design_upper_bound = []
        self.mat_design_lower_bound = []
        self.mat_reward = []
        self.mat_iteration = []
        self.design_process = False  # Initialize to False
        self.mat_file_name = model_name
        self.save_recorded_data = n_steps_train * n_envs_train * 1
        self.reduce_batch_size = n_steps_train * n_envs_train * 1

        self.state = 'hebo_init'  # random or gaussian or init_gaussian or hebo or init_hebo

        self.sampling_hebo_time_period = 100
        self.sampling_gauss_time_period = self.sampling_hebo_time_period // 2
        self.sampling_random_time_period = 0
        self.sampling_hebo_ctr = 0
        self.sampling_gauss_ctr = 0
        self.sampling_random_ctr = 0
        self.hebo_design_history = []
        self.hebo_reward_history = []
        self.percentile_top_designs = 20
        self.batch_size_hebo = 1
        self.batch_size_gauss = 1
        self.batch_size_random = 1
        self.optimal_components = 1
        self.optimal_gmm = None
        self.hebo_init_ctr = 0
        self.gauss_init_ctr = 0

        self.episode_rewards = {}
        self.episode_length = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
        self.best_design = []
        self.best_design_reward = []
        self.target_pos_param = []
        self.init_joint_pos = []
        self.logger_reward = []
        self.logger_reward_prev = -1000
        self.logger_episode_length = []

        self.arm_length_range = [0.1, 1.0]
        self.thruster_radius_range = [0.1, 1.0]


        self.design_space_lb = np.array(
            [self.arm_length_range[0], self.thruster_radius_range[0]])
        self.design_space_ub = np.array(
            [self.arm_length_range[1], self.thruster_radius_range[1]])

        self.design_space_history = []
        self.best_design_gauss = np.array([1000.0, 1000.0])
        self.best_design_reward_gauss = -1000
        self.mat_best_reward_policy = -1000
        self.n_components_range = range(1, 8)
        self.sample_half_gauss = False
        self.design_params = np.ones(8) * 0.5
        self.suggested_design_params = np.ones(2) * 0.5
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/"
        self.column_names = ['x1', 'x2']
        self.rec_gauss_his = pd.DataFrame(columns=self.column_names)
        self.scores_gauss_his = []

        np.set_printoptions(precision=3)

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        print("Training started")

    def _on_rollout_start(self) -> bool:

        if self.state == "hebo_init":
            print("initialising HEBO")

            space_config = []
            self.opt = None

            for i in range(1, len(self.design_space_lb) + 1):
                space_config.append({
                    'name': f'x{i}',
                    'type': 'num',
                    'lb': self.design_space_lb[i - 1],
                    'ub': self.design_space_ub[i - 1]
                })
            space = DesignSpace().parse(space_config)
            self.opt = HEBO(space)
            self.hebo_init_ctr += 1
            self.state = "hebo"

        if self.state == "gauss_init":
            print("initialising gaussian mixture models")
            sorted_indices = []
            top_indices = []
            self.top_designs = []
            self.top_rewards = []
            top_designs_scaled = []

            # decrease the time period for sampling from hebo
            # self.sampling_hebo_time_period -= 5
            # if self.sampling_hebo_time_period <= 30:
            #     self.sampling_hebo_time_period = 30

            self.sampling_gauss_time_period = self.sampling_hebo_time_period // 2

            # define the gaussian mixture models based on the history of hebo
            # order top performing designs based on reward in descending order
            self.hebo_design_history = np.array(self.hebo_design_history)
            self.hebo_reward_history = np.concatenate(self.hebo_reward_history, axis=0)
            sorted_indices = np.argsort(self.hebo_reward_history)[::-1]

            # take the top percentile of the designs
            top_indices = sorted_indices[:int((self.percentile_top_designs / 100) * len(sorted_indices))]

            # keep designs greater than the average reward
            avg_reward = np.mean(self.hebo_reward_history)
            # top_indices = [i for i in range(len(self.hebo_reward_history)) if self.hebo_reward_history[i] > avg_reward]

            print(f"avg reward: {avg_reward}, top indices: {len(top_indices)}")
            print(
                f"top designs: {self.hebo_design_history[top_indices]}, top rewards: {self.hebo_reward_history[top_indices]}")
            self.top_designs = self.hebo_design_history[top_indices]
            self.top_rewards = self.hebo_reward_history[top_indices]
            # find the lower and upper bound of the top designs for each dimension

            self.design_space_lb = np.min(self.top_designs, axis=0)
            self.design_space_ub = np.max(self.top_designs, axis=0)

            self.design_space_history.append([self.design_space_lb, self.design_space_ub])
            print(f"Design space lower bound: {self.design_space_lb}")
            print(f"Design space upper bound: {self.design_space_ub}")
            print(f"Design space history: {self.design_space_history}")
            self.mat_design_upper_bound.append(self.design_space_ub)
            self.mat_design_lower_bound.append(self.design_space_lb)
            # scale self.top_designs to the design space bounded by lb and ub
            top_designs_scaled = (self.top_designs - self.design_space_lb) / (
                        self.design_space_ub - self.design_space_lb)
            # normalise top rewards
            top_rewards_norm = (self.top_rewards - np.min(self.top_rewards)) / (
                        np.max(self.top_rewards) - np.min(self.top_rewards))

            bic_scores = []
            aic_scores = []

            # Fit GMMs with different number of components
            for n_components in self.n_components_range:
                gmm = GaussianMixture(n_components=n_components, random_state=42,
                                      covariance_type='full', init_params='kmeans',
                                      max_iter=100, n_init=20)
                # fit until convergence
                gmm.fit(top_designs_scaled, top_rewards_norm)
                print(f"gmm {n_components} converged: {gmm.converged_}, with iterations: {gmm.n_iter_}")

                # gmm.fit(top_designs_scaled)
                bic_scores.append(gmm.bic(top_designs_scaled))
                aic_scores.append(gmm.aic(top_designs_scaled))

                # Choosing the optimal number of components
                self.optimal_components = np.argmin(bic_scores) + 1  # +1 because range starts from 1
                print(f"Optimal number of components according to BIC: {self.optimal_components}")

            # Fit GMM with the optimal number of components
            self.optimal_gmm = GaussianMixture(n_components=self.optimal_components, random_state=42,
                                               covariance_type='full', init_params='kmeans',
                                               max_iter=100, n_init=20)

            # fit gaussian mixture models
            self.optimal_gmm.fit(top_designs_scaled, top_rewards_norm)

            print(f"GMM means: {self.optimal_gmm.means_}")
            print(f"GMM covariances: {self.optimal_gmm.covariances_}")
            print(
                f"GMM means orignal scale: {self.optimal_gmm.means_ * (self.design_space_ub - self.design_space_lb) + self.design_space_lb}")

            self.hebo_design_history = []
            self.hebo_reward_history = []

            # design_plot = self.optimal_gmm.sample(400)
            # clusters = self.optimal_gmm.predict(design_plot[0])

            # design_plot_ds = design_plot[0] * (self.design_space_ub - self.design_space_lb) + self.design_space_lb
            # # Plotting the clusters in 3D
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # scatter = ax.scatter(design_plot_ds[:, 0], design_plot_ds[:, 1], design_plot_ds[:, 2], c=clusters, cmap='viridis', marker='o')
            # legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
            # ax.add_artist(legend1)
            # ax.set_title("Gaussian Mixture Model Clustering on Complex 3D Data")
            # ax.set_xlabel("X axis")
            # ax.set_ylabel("Y axis")
            # ax.set_zlabel("Z axis")
            # ax.set_xlim(0.1, 2.0)
            # ax.set_ylim(0.1, 2.0)
            # ax.set_zlim(0.1, 2.0)
            # plt.show()

            self.gauss_init_ctr += 1

            self.state = "gauss"

        if self.state == "hebo":

            # turn off train when sampling from hebo
            self.model.evaluate_current_policy = True
            print("...Updating the new lengths...")
            start_time = time.time()

            if not self.sample_half_gauss:
                # sample from hebos
                self.rec = self.opt.suggest(n_suggestions=(self.n_envs_train // self.batch_size_hebo))
            else:
                # sample half from hebos
                self.rec = self.opt.suggest(n_suggestions=(self.n_envs_train // (self.batch_size_hebo * 2)))
                # #sample half from gaussian mixture models
                # gauss_output_scaled = self.optimal_gmm.sample(self.n_envs_train // (self.batch_size_hebo*2))

                # #scale the sampled designs to the design space bounded by lb and ub
                # gauss_output = gauss_output_scaled[0] * (self.design_space_ub - self.design_space_lb) + self.design_space_lb
                # gauss_output = np.clip(gauss_output, self.design_space_lb, self.design_space_ub)

                # # Convert gauss_output[0] to a DataFrame with the same column names
                # gauss_output_df = pd.DataFrame(gauss_output, columns=self.column_names)

                prev_top_designs = pd.DataFrame(self.top_designs[:int(self.n_envs_train // (self.batch_size_hebo * 2))],
                                                columns=self.column_names)

                # Now concatenate the DataFrames
                self.rec = pd.concat([self.rec, prev_top_designs], ignore_index=True)
                print("first hebo", self.rec)
                self.sample_half_gauss = False

            for i in range(self.n_envs_train // self.batch_size_hebo):
                for j in range(self.batch_size_hebo):
                    self.suggested_design_params = self.rec.values[i]
                    self.design_params = np.array(
                        [self.suggested_design_params[0], self.suggested_design_params[0], self.suggested_design_params[0],
                         self.suggested_design_params[0], self.suggested_design_params[1], self.suggested_design_params[1],
                         self.suggested_design_params[1], self.suggested_design_params[1]])
                    self.modify_xml_quadcopter_full_geometry(f"{self.mujoco_file_folder}quadcopter_{i}.xml", self.design_params)
                    self.training_env.env_method('__init__', i, indices=[i * self.batch_size_hebo + j])
                    self.training_env.env_method('set_design_params', self.design_params,
                                                 indices=[i * self.batch_size_hebo + j])
                    self.training_env.env_method('reset', indices=[i * self.batch_size_hebo + j])

            print(f"Design proposal took {time.time() - start_time:.2f} seconds")
            self.sampling_hebo_ctr += 1

        if self.state == "gauss":

            # turn on train when sampling from gaussian mixture models
            self.model.evaluate_current_policy = False

            # sample from gaussian mixture models
            self.gauss_output = self.optimal_gmm.sample(self.n_envs_train // self.batch_size_gauss)
            self.gauss_designs = self.gauss_output[0]

            # scale the sampled designs to the design space bounded by lb and ub
            self.gauss_designs = self.gauss_designs * (
                        self.design_space_ub - self.design_space_lb) + self.design_space_lb
            self.gauss_designs = np.clip(self.gauss_designs, self.design_space_lb, self.design_space_ub)
            # sample from gaussian mixture models and set the designs in the environments
            for i in range(self.n_envs_train // self.batch_size_gauss):
                for j in range(self.batch_size_gauss):
                    gauss_initial = self.gauss_designs[i]
                    gauss_modified = np.array(
                        [gauss_initial[0], gauss_initial[0], gauss_initial[0], gauss_initial[0], gauss_initial[1],
                         gauss_initial[1], gauss_initial[1], gauss_initial[1], gauss_initial[1]])
                    self.modify_xml_quadcopter_full_geometry(f"{self.mujoco_file_folder}quadcopter_{i}.xml", gauss_modified)
                    self.training_env.env_method('__init__', i, indices=[i * self.batch_size_gauss + j])
                    self.training_env.env_method('set_design_params', gauss_modified,
                                                 indices=[i * self.batch_size_gauss + j])
                    self.training_env.env_method('reset', indices=[i * self.batch_size_gauss + j])

            self.sampling_gauss_ctr += 1
            self.sample_half_gauss = True

        return True

    def _on_rollout_end(self) -> bool:

        if self.state == "hebo":
            print("...Starting design distribution update...")

            start_time = time.time()

            scores = []

            self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
            self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]

            for i in range(self.n_envs_train // self.batch_size_hebo):
                # average batch reward
                sum_reward = 0
                total_episode_length = 0
                for j in range(self.batch_size_hebo):
                    sum_reward += self.episode_rewards[i * self.batch_size_hebo + j] / self.design_iteration[
                        i * self.batch_size_hebo + j]
                    total_episode_length += self.episode_length[i * self.batch_size_hebo + j] / self.design_iteration[
                        i * self.batch_size_hebo + j]

                self.design_rewards_avg[i] = sum_reward / self.batch_size_hebo
                self.episode_length_avg[i] = total_episode_length / self.batch_size_hebo
                score_array = np.array(self.design_rewards_avg[i]).reshape(-1, 1)  # Convert to NumPy array
                scores.append(-score_array)  # HEBO minimizes, so we need to negate the scores

                # Logging
                current_design_params = \
                self.training_env.env_method('get_design_params', indices=[i * self.batch_size_hebo])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.batch_size_hebo])[0]
                # print(f"Env ID: {dist_env_id}, mean reward: {self.design_rewards_avg[i]}, Mean episode length: {self.episode_length_avg[i]}, arm length: {current_limb_length}")
                self.logger.record("mean reward", self.design_rewards_avg[i])
                self.logger.record("mean episode length", self.episode_length_avg[i])

                # Matlab logging
                self.mat_design_params.append(current_design_params)
                self.mat_reward.append(self.design_rewards_avg[i])
                self.mat_iteration.append(self.episode_length_avg[i])

                self.logger_reward.append(self.design_rewards_avg[i])
                self.logger_episode_length.append(self.episode_length_avg[i])

                if self.design_rewards_avg[i] > self.mat_best_reward_policy:
                    self.mat_best_reward_policy = self.design_rewards_avg[i]
                    self.model.save(
                        f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/bestHeboDesign_{self.model_name}")

            # Logging
            self.logger.record("mean reward", np.mean(self.logger_reward))
            self.logger.record("mean episode length", np.mean(self.logger_episode_length))
            self.logger_reward = []
            self.logger_episode_length = []

            scores = np.array(scores)  # Make sure the outer list is also a NumPy array
            # save history of design and reward proposed by hebo
            self.hebo_design_history.extend(self.rec.values.tolist())
            self.hebo_reward_history.extend(np.concatenate(-scores).tolist())

            # Update the design distribution
            scores = scores.reshape(-1, 1)
            self.opt.observe(self.rec, scores)

            # #check if self.rec_gauss_his is empty
            # if len(self.scores_gauss_his) > 0:
            #     self.scores_gauss_his = np.array(self.scores_gauss_his).reshape(-1, 1)
            #     self.opt.observe(self.rec_gauss_his, self.scores_gauss_his)
            #     self.rec_gauss_his = pd.DataFrame(columns=self.column_names)
            #     self.scores_gauss_his = []

            # After all iterations, print the best input and output
            best_idx = self.opt.y.argmin()
            best_design = self.opt.X.iloc[best_idx]
            best_design_reward = self.opt.y[best_idx]

            print(f"Best design: {best_design}, best reward: {best_design_reward}")
            self.best_design.append(best_design)
            self.best_design_reward.append(best_design_reward)

            print(f"Design distribution update took {time.time() - start_time:.2f} seconds")

            # Reset episode reward accumulator
            self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
            self.design_iteration = [1 for _ in range(self.n_envs_train)]
            self.episode_rewards = {}
            self.episode_length = {}

        if self.state == "gauss":
            # calculate the reward for the sampled designs
            self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_gauss)]
            self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_gauss)]

            for i in range(self.n_envs_train // self.batch_size_gauss):
                # average batch reward
                sum_reward = 0
                total_episode_length = 0
                for j in range(self.batch_size_gauss):
                    sum_reward += self.episode_rewards[i * self.batch_size_gauss + j] / self.design_iteration[
                        i * self.batch_size_gauss + j]
                    total_episode_length += self.episode_length[i * self.batch_size_gauss + j] / self.design_iteration[
                        i * self.batch_size_gauss + j]

                self.design_rewards_avg[i] = sum_reward / self.batch_size_gauss
                self.episode_length_avg[i] = total_episode_length / self.batch_size_gauss
                self.logger_reward.append(self.design_rewards_avg[i])
                self.logger_episode_length.append(self.episode_length_avg[i])
                current_design_params = \
                self.training_env.env_method('get_design_params', indices=[i * self.batch_size_gauss])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.batch_size_gauss])[0]
                # print(f"Env ID: {dist_env_id}, mean reward: {self.design_rewards_avg[i]}, Mean episode length: {self.episode_length_avg[i]}, arm length: {current_limb_length}")

                # # append current_limb_length in DataFrame with the same column names
                # current_limb_length_reshaped = current_limb_length.reshape(1, -1)  # Reshapes to (1, 6)
                # # Append current_limb_length as a new row in DataFrame with the correct column names
                # self.rec_gauss_his = pd.concat([self.rec_gauss_his, pd.DataFrame(current_limb_length_reshaped, columns=self.column_names)], ignore_index=True)
                # self.scores_gauss_his.append(-self.design_rewards_avg[i])

                # find the best design
                if self.design_rewards_avg[i] > self.best_design_reward_gauss:
                    self.best_design_reward_gauss = self.design_rewards_avg[i]
                    self.best_design_gauss = current_design_params
                    self.mat_best_design_gauss.append(current_design_params)
                    self.model.save(
                        f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/bestGaussDesign_{self.model_name}")

            # Logging
            self.logger.record("mean reward", np.mean(self.logger_reward))
            self.logger.record("mean episode length", np.mean(self.logger_episode_length))
            self.logger_reward = []
            self.logger_episode_length = []

            # save the best model
            # if np.mean(self.design_rewards_avg) >= self.logger_reward_prev:
            # self.logger_reward_prev = np.mean(self.design_rewards_avg)
            self.model.save(
                f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/{self.model_name}")
            # print(f"Model saved, reward: {self.logger_reward_prev}, iteration: {self.gauss_init_ctr}, best design: {self.best_design_gauss}")

            # Reset episode reward accumulator
            self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
            self.design_iteration = [1 for _ in range(self.n_envs_train)]
            self.episode_rewards = {}
            self.episode_length = {}

        output_data = {
            "design_params": np.array(self.mat_design_params),
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_iteration),
            "best_design": np.array(self.mat_best_design_gauss),
            "design_space_lb": np.array(self.mat_design_lower_bound),
            "design_space_ub": np.array(self.mat_design_upper_bound),
        }
        print("saving matlab data...")
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/rl/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)

        if self.sampling_hebo_ctr >= self.sampling_hebo_time_period:
            self.state = "gauss_init"
            self.sampling_hebo_ctr = 0

        elif self.sampling_gauss_ctr >= self.sampling_gauss_time_period:
            self.state = "hebo_init"
            self.sampling_gauss_ctr = 0

        return True

    def _on_step(self) -> bool:

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done:
                    self.design_iteration[i] += 1
        return True


    def modify_xml_quadcopter_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_paramss: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        [arm0, arm1, arm2, arm3, thruster0, thruster1, thruster2, thruster3] = design_params

        arms = root.findall(".//geom[@name='arm0']")
        for arm in arms:
            new_size = [str(arm0)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm1']")
        for arm in arms:
            new_size = [str(arm1)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm2']")
        for arm in arms:
            new_size = [str(arm2)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm3']")
        for arm in arms:
            new_size = [str(arm3)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))

        thrusters = root.findall(".//geom[@name='thruster0']")
        for thruster in thrusters:
            new_size = [str(thruster0)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))
        thrusters = root.findall(".//geom[@name='thruster1']")
        for thruster in thrusters:
            new_size = [str(thruster1)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster2']")
        for thruster in thrusters:
            new_size = [str(thruster2)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster3']")
        for thruster in thrusters:
            new_size = [str(thruster3)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        motors = root.findall(".//site[@name='motor0']")
        for motor in motors:
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor1']")
        for motor in motors:
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor2']")
        for motor in motors:
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        motors = root.findall(".//site[@name='motor3']")
        for motor in motors:
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            motor.set('pos', ' '.join(new_pos))
        tree.write(file_path)


class evaluate_design(BaseCallback):
    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 10, n_envs_train=8, verbose=0):

        super(evaluate_design, self).__init__(verbose)
        self.model = model
        self.n_envs_train = n_envs_train
        self.n_steps_train = n_steps_train
        self.episode_rewards = {}
        self.rewards_iteration = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards = [0 for _ in range(self.n_envs_train)]
        self.episode_length = {}
        self.mat_design_params = []
        self.mat_reward = []
        self.mat_iteration = []
        self.average_reward = []
        self.average_episode_length = []
        self.model_name = model_name
        self.mat_file_name = model_name
        self.design_iteration = [0 for _ in range(self.n_envs_train)]
        self.model.evaluate_current_policy = True
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/"
        self.design_params = np.ones(8) * 0.5

    def _on_rollout_start(self) -> bool:

        # reset the environments
        for i in range(self.n_envs_train):
            self.arm_1 = 0.1
            self.arm_2 = 0.1
            self.arm_3 = 0.1
            self.arm_4 = 0.1
            self.thruster_1 = 0.05
            self.thruster_2 = 0.05
            self.thruster_3 = 0.05
            self.thruster_4 = 0.05

            self.design_params = np.array([self.arm_1, self.arm_2, self.arm_3, self.arm_4, self.thruster_1, self.thruster_2, self.thruster_3, self.thruster_4])

            self.modify_xml_quadcopter_full_geometry(f"{self.mujoco_file_folder}quadcopter_{i}.xml", self.design_params)
            self.training_env.env_method('__init__', i, indices=[i])
            self.training_env.env_method("set_design_params", self.design_params, indices=[i])
            self.training_env.env_method('reset', indices=[i])
        return True


    def _on_step(self) -> bool:

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done or self.episode_length[i] >= self.n_steps_train:
                    # current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
                    # target_pos_tcp = self.training_env.env_method('get_target_pos_tcp', indices=[i])[0]
                    # dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                    self.average_episode_length.append(self.episode_length[i])
                    self.average_reward.append(self.episode_rewards[i])
                    self.design_iteration[i] += 1
                    # Reset episode reward accumulator
                    self.episode_rewards[i] = 0
                    self.episode_length[i] = 0


        return True

    def _on_rollout_end(self) -> bool:

        for i in range(self.n_envs_train):
            current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
            # Matlab logging
            self.mat_design_params.append(current_design_params)
            self.mat_reward.append(self.episode_rewards[i])
            self.mat_iteration.append(self.episode_length[i])
        self.logger.record("mean episode length", np.sum(self.average_episode_length) / np.sum(self.design_iteration))
        self.logger.record("mean reward", np.sum(self.average_reward) / np.sum(self.design_iteration))


        output_data = {
            "design_params": np.array(self.mat_design_params),
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_iteration),

        }
        print("saving matlab data...")
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        self.average_episode_length = []
        self.average_reward = []
        self.design_iteration = [0 for _ in range(self.n_envs_train)]

        return True

    def modify_xml_quadcopter_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_paramss: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        original_lengths = {
            'arm': 0.05,
            'thruster': 0.05
        }


        [arm0, arm1, arm2, arm3, thruster0, thruster1, thruster2, thruster3] = design_params


        arms = root.findall(".//geom[@name='arm0']")
        for arm in arms:
            new_size = [str(arm0)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm1']")
        for arm in arms:
            new_size = [str(arm1)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm2']")
        for arm in arms:
            new_size = [str(arm2)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))
        arms = root.findall(".//geom[@name='arm3']")
        for arm in arms:
            new_size = [str(arm3)] + [str(0.01)] + [str(0.0025)]
            arm.set('size', ' '.join(new_size))

        thrusters = root.findall(".//geom[@name='thruster0']")
        for thruster in thrusters:
            new_size = [str(thruster0)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm0)] + [str(arm0)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))
        thrusters = root.findall(".//geom[@name='thruster1']")
        for thruster in thrusters:
            new_size = [str(thruster1)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(arm1)] + [str(-arm1)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster2']")
        for thruster in thrusters:
            new_size = [str(thruster2)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm2)] + [str(-arm2)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))

        thrusters = root.findall(".//geom[@name='thruster3']")
        for thruster in thrusters:
            new_size = [str(thruster3)] + [str(0.0025)]
            thruster.set('size', ' '.join(new_size))
            new_pos = [str(-arm3)] + [str(arm3)] + [str(0)]
            thruster.set('pos', ' '.join(new_pos))


        motors = root.findall(".//motor[@site='motor']")
        for motor in motors:
            new_gear = [str(0)] + [str(0)] + [str(arm0 * thruster0 / (original_lengths['arm'] * original_lengths['thruster']))] + [str(arm0 * arm0 * thruster0 / (original_lengths['arm'] * original_lengths['arm'] * original_lengths['thruster']))] + [str(arm0 * arm0 * thruster0 / (original_lengths['arm'] * original_lengths['arm'] * original_lengths['thruster']))] + [str(arm0 * arm0 * thruster0 / (original_lengths['arm'] * original_lengths['arm'] * original_lengths['thruster']))]
            motor.set('gear', ' '.join(new_gear))
        tree.write(file_path)


if __name__ == '__main__':
    main()