﻿import time
from typing import Any, Dict
import numpy as np
import gym
import pandas as pd
import os
import shutil
import xml.etree.ElementTree as ET
from stable_baselines3 import PPO
from drone_v1 import DroneEnv
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

def main():
    #training parameters
    use_sde = False
    hidden_sizes_train = 256
    REWARD = np.array([1.0, 0.0])
    learning_rate_train = 0.0001
    n_epochs_train = 10
    LOAD_OLD_MODEL = True
    n_steps_train = 512 * 10
    n_envs_train = 64
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 10000

    batch_size_train = 128
    global_iteration = 0
    TRAIN = False
    CALL_BACK_FUNC = f"evaluate_design"

    original_xml_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/assets/drone.xml"
    destination_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/assets/"

    for i in range(n_envs_train):
        new_file_name = f"drone_{i}.xml"
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


        env_fns = [lambda config=config: DroneEnv(**config) for config in env_configs]

        vec_env = SubprocVecEnv(env_fns, start_method='fork')

        n_envs_eval = 1
        env_configs_eval = [{'env_id': i, 'ctrl_cost_weight': 0.5, 'render_mode': 'human'} for i in range(n_envs_eval)]

        assert len(env_configs_eval) == n_envs_eval

        env_fns_eval = [lambda config=config: DroneEnv(**config) for config in env_configs_eval]

        vec_env_eval = DummyVecEnv(env_fns_eval)


        model_name = f"drone_constant_design"
        log_dir = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/drone_tensorboard/TB_{model_name}"

        if LOAD_OLD_MODEL is True:
            new_model = []
            old_model = PPO.load(f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/trained_model/drone_constant_design_3.zip", env = vec_env)

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
        elif CALL_BACK_FUNC is f"evaluate_design":
            param_changer = evaluate_design(model_name = model_name, model = new_model_eval, n_steps_train = n_steps_train, n_envs_train = n_envs_eval, verbose=1)
        else:
            print("No callback function specified")
            break


        if TRAIN is True:
            new_model.learn(total_timesteps = total_timesteps_train ,progress_bar=True, callback=param_changer)
            print("Model trained, saving...")
            new_model.save(f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/trained_model/{model_name}")
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
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/assets/"

        self.design_params = np.ones(2)

    def _on_rollout_start(self) -> bool:

        # reset the environments
        for i in range(self.n_envs_train):
            self.thickness = 0.01
            self.semispan = 2.0
            self.chord = 0.5

            self.design_params = np.array([self.thickness, self.semispan, self.chord])

            self.modify_xml_drone_full_geometry(f"{self.mujoco_file_folder}drone_{i}.xml", self.design_params)
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
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/trained_model/{self.model_name}")


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
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        self.average_episode_length = []
        self.average_reward = []
        self.design_iteration = [0 for _ in range(self.n_envs_train)]

        return True

    def modify_xml_drone_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_paramss: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()


        thickness = design_params[0]
        semispan = design_params[1]
        chord = design_params[2]


        element_body_names = ['left_wing', 'right_wing']
        element_geom_names = ['left_wing_geom', 'right_wing_geom']



        left_wing_geom = root.findall(".//geom[@name='left_wing_geom']")
        for geom in left_wing_geom:
            current_size = geom.get('size').split(' ')
            new_size = [str(thickness)] + [str(semispan)] + [str(chord)]
            geom.set('size', ' '.join(new_size))

        left_wing = root.findall(".//body[@name='left_wing']")
        for body in left_wing:
            current_pos = body.get('pos').split(' ')
            new_pos = [str(0)] + [str(-semispan)] + [str(0)]
            body.set('pos', ' '.join(new_pos))

        right_wing_geom = root.findall(".//geom[@name='right_wing_geom']")
        for geom in right_wing_geom:
            current_size = geom.get('size').split(' ')
            new_size = [str(thickness)] + [str(semispan)] + [str(chord)]
            geom.set('size', ' '.join(new_size))

        right_wing = root.findall(".//body[@name='right_wing']")
        for body in right_wing:
            current_pos = body.get('pos').split(' ')
            new_pos = [str(0)] + [str(semispan)] + [str(0)]
            body.set('pos', ' '.join(new_pos))

        joints = root.findall(f".//joint[@name='left_wing_x_joint']")
        for joint in joints:
            joint_pos = [0, semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))
        joints = root.findall(f".//joint[@name='left_wing_y_joint']")
        for joint in joints:
            joint_pos = [0, semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))
        joints = root.findall(f".//joint[@name='left_wing_z_joint']")
        for joint in joints:
            joint_pos = [0, semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))

        joints = root.findall(f".//joint[@name='right_wing_x_joint']")
        for joint in joints:
            joint_pos = [0, -semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))
        joints = root.findall(f".//joint[@name='right_wing_y_joint']")
        for joint in joints:
            joint_pos = [0, -semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))
        joints = root.findall(f".//joint[@name='right_wing_z_joint']")
        for joint in joints:
            joint_pos = [0, -semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))

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
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/drone/assets/"
        self.design_params = np.ones(3) * 0.5

    def _on_rollout_start(self) -> bool:

        # reset the environments
        for i in range(self.n_envs_train):
            self.thickness = 0.01
            self.semispan = 2.0
            self.chord = 0.5
            self.design_params= np.array([self.thickness, self.semispan, self.chord])

            self.modify_xml_drone_full_geometry(f"{self.mujoco_file_folder}drone_{i}.xml", self.design_params)
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

    def modify_xml_drone_full_geometry(self, file_path, design_params):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - design_params: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        thickness = design_params[0]
        semispan = design_params[1]
        chord = design_params[2]

        element_body_names = ['left_wing', 'right_wing']
        element_geom_names = ['left_wing_geom', 'right_wing_geom']

        left_wing_geom = root.findall(".//geom[@name='left_wing_geom']")
        for geom in left_wing_geom:
            current_size = geom.get('size').split(' ')
            new_size = [str(thickness)] + [str(semispan)] + [str(chord)]
            geom.set('size', ' '.join(new_size))

        left_wing = root.findall(".//body[@name='left_wing']")
        for body in left_wing:
            current_pos = body.get('pos').split(' ')
            new_pos = [str(0)] + [str(-semispan - 0.1)] + [str(0)]
            body.set('pos', ' '.join(new_pos))

        right_wing_geom = root.findall(".//geom[@name='right_wing_geom']")
        for geom in right_wing_geom:
            current_size = geom.get('size').split(' ')
            new_size = [str(thickness)] + [str(semispan)] + [str(chord)]
            geom.set('size', ' '.join(new_size))

        right_wing = root.findall(".//body[@name='right_wing']")
        for body in right_wing:
            current_pos = body.get('pos').split(' ')
            new_pos = [str(0)] + [str(semispan + 0.1)] + [str(0)]
            body.set('pos', ' '.join(new_pos))

        joints = root.findall(f".//joint[@name='left_wing_x_joint']")
        for joint in joints:
            joint_pos = [0, semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))
        joints = root.findall(f".//joint[@name='left_wing_y_joint']")
        for joint in joints:
            joint_pos = [0, semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))
        joints = root.findall(f".//joint[@name='left_wing_z_joint']")
        for joint in joints:
            joint_pos = [0, semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))

        joints = root.findall(f".//joint[@name='right_wing_x_joint']")
        for joint in joints:
            joint_pos = [0, -semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))
        joints = root.findall(f".//joint[@name='right_wing_y_joint']")
        for joint in joints:
            joint_pos = [0, -semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))
        joints = root.findall(f".//joint[@name='right_wing_z_joint']")
        for joint in joints:
            joint_pos = [0, -semispan, 0]
            joint.set('pos', ' '.join(map(str, joint_pos)))

        tree.write(file_path)


if __name__ == '__main__':
    main()