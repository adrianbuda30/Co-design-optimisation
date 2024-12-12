import time
from typing import Any, Dict
import numpy as np
import gym
import pandas as pd
import os
import shutil
import xml.etree.ElementTree as ET
from stable_baselines3 import PPO
from ant_v4 import AntEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from scipy.io import loadmat, savemat
import random
import torch
import math 
from sklearn.mixture import GaussianMixture



def main():
    #training parameters
    use_sde = False
    hidden_sizes_train = 256
    REWARD = np.array([1.0, 0.0])
    learning_rate_train = 0.0005
    n_epochs_train = 10
    LOAD_OLD_MODEL = False
    n_steps_train = 512 * 2
    n_envs_train = 64
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 10000

    batch_size_train = 128
    global_iteration = 0
    TRAIN = True
    CALL_BACK_FUNC = f"Schaff_callback"

    original_xml_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/assets/ant.xml"
    destination_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/assets/"
    # Copy the original file with new names
    for i in range(n_envs_train):
        new_file_name = f"ant_{i}.xml"
        new_file_path = os.path.join(destination_folder, new_file_name)
        shutil.copy2(original_xml_path, new_file_path)
        # print(f"Copied to: {new_file_path}")

    while True:
        # initialise the model PPO
        learning_rate_train = learning_rate_train

        onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                               net_arch=dict(vf=[hidden_sizes_train, hidden_sizes_train],
                                             pi=[hidden_sizes_train, hidden_sizes_train]))

        global_iteration += 1

        # Define unique initialization variables for each environment
        env_configs = [{'env_id': i, 'ctrl_cost_weight': 0.5} for i in range(n_envs_train)]
        # Ensure we have configurations for each environment instance
        assert len(env_configs) == n_envs_train

        # Create function for each environment instance with its unique configuration
        env_fns = [lambda config=config: AntEnv(**config) for config in env_configs]
        # Create the vectorized environment using SubprocVecEnv directly
        vec_env = SubprocVecEnv(env_fns, start_method='fork')

        # Define unique initialization variables for each evaluation environment
        n_envs_eval = 1  # For visualization, using a single environment is simpler
        env_configs_eval = [{'env_id': i, 'ctrl_cost_weight': 0.5, 'render_mode': 'human'} for i in
                            range(n_envs_eval)]

        # Ensure we have configurations for each evaluation environment instance
        assert len(env_configs_eval) == n_envs_eval

        # Create function for each evaluation environment instance with its unique configuration
        env_fns_eval = [lambda config=config: AntEnv(**config) for config in env_configs_eval]

        # Create the vectorized environment using DummyVecEnv for evaluation
        vec_env_eval = DummyVecEnv(env_fns_eval)

        model_name = f"ant_Schaff_1distrib_trial_25params"
        log_dir = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/ant_tensorboard/TB_{model_name}"

        if LOAD_OLD_MODEL is True:
            new_model = []
            old_model = PPO.load(
                f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/rl/trained_model/bestHeboDesign_ant_random_1.zip",
                env=vec_env)
            # Create a new model with the desired configuration
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

            # Load the weights from the old model
            new_model.set_parameters(old_model.get_parameters())
            new_model_eval.set_parameters(old_model.get_parameters())

        else:
            new_model = PPO("MlpPolicy", env=vec_env, n_steps=n_steps_train, batch_size=batch_size_train,
                            n_epochs=n_epochs_train, use_sde=use_sde, ent_coef=entropy_coeff_train,
                            learning_rate=learning_rate_train,
                            policy_kwargs=onpolicy_kwargs, device='cpu', verbose=1, tensorboard_log=log_dir)
            print("New model created")

        # Train the new model
        print("Model training...")
        # Now you can continue training with the new model
        if CALL_BACK_FUNC is f"random_design":
            param_changer = random_design(model_name=model_name, model=new_model, n_steps_train=n_steps_train,
                                          n_envs_train=n_envs_train, verbose=1)

        elif CALL_BACK_FUNC is f"Schaff_callback":
            param_changer = Schaff_callback(model_name=model_name, model=new_model, n_steps_train = n_steps_train, n_envs_train=n_envs_train, num_distributions=1, verbose=1)
        elif CALL_BACK_FUNC is f"Schaff_callback_GMM":
            param_changer = Schaff_callback_GMM(model_name=model_name, model=new_model, n_steps_train=n_steps_train, n_envs_train=n_envs_train, num_distributions=64, verbose=1)
        elif CALL_BACK_FUNC is f"Hebo_Gauss_callback":
            param_changer = Hebo_Gauss_callback(model_name=model_name, model=new_model, n_steps_train=n_steps_train,
                                                n_envs_train=n_envs_train, verbose=1)
        elif CALL_BACK_FUNC is f"evaluate_design":
            param_changer = evaluate_design(model_name=model_name, model=new_model_eval,
                                            n_steps_train=n_steps_train, n_envs_train=n_envs_eval, verbose=1)
        else:
            print("No callback function specified")
            break

        if TRAIN is True:
            new_model.learn(total_timesteps=total_timesteps_train, progress_bar=True, callback=param_changer)
            print("Model trained, saving...")
            new_model.save(
                f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/rl/hebo/{model_name}")
            print("Model saved")
            LOAD_OLD_MODEL = True
            vec_env.close()
        else:
            new_model_eval.learn(total_timesteps=total_timesteps_train, progress_bar=True, callback=param_changer)
            print("Model trained, saving...")
            LOAD_OLD_MODEL = True
            vec_env_eval.close()

        break


class random_design(BaseCallback):
    def __init__(self, model_name = f"matfile", model = None, n_steps_train=512 * 10, n_envs_train=8, verbose=0):

        super(random_design, self).__init__(verbose)
        self.model = model
        self.n_envs_train = n_envs_train
        self.n_steps_train = n_steps_train
        self.episode_rewards = {}
        self.rewards_iteration = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards = [0 for _ in range(self.n_envs_train)]
        self.episode_length = {}
        self.mat_limb_length = []
        self.mat_reward = []
        self.mat_iteration = []
        self.mat_file_name = model_name
        self.average_reward = []
        self.average_episode_length = []
        self.model_name = model_name
        self.design_iteration = [0 for _ in range(self.n_envs_train)]
        self.my_custom_condition = True  # Initialize your condition
        self.model.evaluate_current_policy = False
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/assets/"
        self.limb_radius_range = [0.01, 0.4]
        self.limb_length_range = [0.4, 2.0]
        self.torus_radius_range = [0.01, 0.4]

        self.limb_length = np.ones(25) * 0.5

    def _on_rollout_start(self) -> bool:

        #reset the environments
        for i in range(self.n_envs_train):
            
            self.limb_length = np.array([random.uniform(self.torus_radius_range[0], self.torus_radius_range[1]), 
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1]),
                                        random.uniform(self.limb_radius_range[0], self.limb_radius_range[1])])
            
            self.modify_xml_ant_full_geometry(f"{self.mujoco_file_folder}ant_{i}.xml", self.limb_length)
            self.training_env.env_method('__init__', i ,indices=[i])
            self.training_env.env_method("set_limb_length", self.limb_length, indices=[i])
            self.training_env.env_method('reset', indices=[i])
            # print(self.training_env.env_method('get_env_id', indices=[i]))

            # print(self.training_env.env_method('get_limb_length', indices=[i]))
        # time.sleep(20)

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
                    # Reset episode reward accumulator
                    self.episode_rewards[i] = 0
                    self.episode_length[i] = 0
        return True
    
    def _on_rollout_end(self) -> bool:
        self.model.save(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/trained_model/random_design/{self.model_name}")
        # Logging

        for i in range(self.n_envs_train):
            current_limb_length = self.training_env.env_method('get_limb_length', indices=[i])[0]
            # Matlab logging
            self.mat_limb_length.append(current_limb_length)
            self.mat_reward.append(self.episode_rewards[i])
            self.mat_iteration.append(self.episode_length[i])
        self.logger.record("mean episode length", np.sum(self.average_episode_length) / np.sum(self.design_iteration))
        self.logger.record("mean reward", np.sum(self.average_reward) / np.sum(self.design_iteration))

        output_data = {
            "limb_length": np.array(self.mat_limb_length),
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_iteration),

        }
        print("saving matlab data...")
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/rl/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        self.average_episode_length = []
        self.average_reward = []
        self.design_iteration = [0 for _ in range(self.n_envs_train)]

        return True
    

    def modify_xml_geometry(self, file_path, limb_lengths):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.
        
        Args:
        - file_path: Path to the XML file to modify.
        - limb_lengths: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        limb_lengths = 0.7071 * limb_lengths
        # Names of the elements to modify
        element_body_names = ['front_left_foot', 'front_right_foot', 'back_foot', 'right_back_foot']
        element_geom_names_last = ['left_ankle_geom', 'right_ankle_geom', 'third_ankle_geom', 'fourth_ankle_geom']
        element_geom_names_first = ['left_leg_geom', 'right_leg_geom', 'back_leg_geom', 'rightback_leg_geom']
        
        # Update 'fromto' for geoms
        for i, name in enumerate(element_geom_names_first):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                new_fromto = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[i]) if float(coord) != 0 else '0' for coord in current_fromto])
                geom.set('fromto', new_fromto)

        for i, name in enumerate(element_geom_names_last):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                index = i + len(element_geom_names_last)  
                new_fromto = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord in current_fromto])
                geom.set('fromto', new_fromto)

        # Update 'pos' for bodies
        for i, name in enumerate(element_body_names):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                # Assuming limb_lengths for bodies start after the last geom
                new_pos = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[i]) if float(coord) != 0 else '0' for coord in current_pos])
                body.set('pos', new_pos)
        
        # Save the modified XML file
        tree.write(file_path)

    def modify_xml_ant_geometry(self, file_path, limb_lengths):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.
        
        Args:
        - file_path: Path to the XML file to modify.
        - limb_lengths: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        limb_lengths = 0.7071 * limb_lengths
        # Names of the elements to modify
        element_body_names = ['front_left_foot', 'front_right_foot', 'back_foot', 'right_back_foot']
        element_geom_names_last = ['left_ankle_geom', 'right_ankle_geom', 'third_ankle_geom', 'fourth_ankle_geom']
        element_geom_names_first = ['left_leg_geom', 'right_leg_geom', 'back_leg_geom', 'rightback_leg_geom']
        element_geom_thigh = ['aux_1_geom', 'aux_2_geom', 'aux_3_geom', 'aux_4_geom']
        element_body_thigh = ['aux_1', 'aux_2', 'aux_3', 'aux_4']

        #set new size for torso
        torso = root.findall(f".//geom[@name='torso_geom']")
        for geom in torso:
            new_size = ' '.join([str(float(limb_lengths[0]))])
            geom.set('size', new_size)

        for i, name in enumerate(element_geom_thigh):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                new_fromto = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[1]) if float(coord) != 0 else '0' for coord in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[2]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        for i, name in enumerate(element_body_thigh):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                # Assuming limb_lengths for bodies start after the last geom
                new_pos = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[1]) if float(coord) != 0 else '0' for coord in current_pos])
                body.set('pos', new_pos)

        # Update 'fromto' for geoms
        for i, name in enumerate(element_geom_names_first):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                new_fromto = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[3]) if float(coord) != 0 else '0' for coord in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[4]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        # Update 'pos' for bodies
        for i, name in enumerate(element_body_names):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                # Assuming limb_lengths for bodies start after the last geom
                new_pos = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[3]) if float(coord) != 0 else '0' for coord in current_pos])
                body.set('pos', new_pos)



        for i, name in enumerate(element_geom_names_last):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                new_fromto = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[5]) if float(coord) != 0 else '0' for coord in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[6]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        # Save the modified XML file
        tree.write(file_path)

    def modify_xml_ant_full_geometry(self, file_path, limb_lengths):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.
        
        Args:
        - file_path: Path to the XML file to modify.
        - limb_lengths: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        limb_lengths = 0.7071 * limb_lengths
        # Names of the elements to modify
        element_body_names = ['front_left_foot', 'front_right_foot', 'back_foot', 'right_back_foot']
        element_geom_names_last = ['left_ankle_geom', 'right_ankle_geom', 'third_ankle_geom', 'fourth_ankle_geom']
        element_geom_names_first = ['left_leg_geom', 'right_leg_geom', 'back_leg_geom', 'rightback_leg_geom']
        element_geom_thigh = ['aux_1_geom', 'aux_2_geom', 'aux_3_geom', 'aux_4_geom']
        element_body_thigh = ['aux_1', 'aux_2', 'aux_3', 'aux_4']
        element_geom_names_aux = ['left_leg_geom_aux', 'right_leg_geom_aux', 'back_leg_geom_aux', 'rightback_leg_geom_aux']

        #set new size for torso
        torso = root.findall(f".//geom[@name='torso_geom']")
        for geom in torso:
            new_size = ' '.join([str(float(limb_lengths[0]))])
            geom.set('size', new_size)

        for i, name in enumerate(element_geom_thigh):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                index = i + 1
                new_fromto = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[index+4]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        for i, name in enumerate(element_body_thigh):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                # Assuming limb_lengths for bodies start after the last geom
                index = i + 1
                new_pos = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord in current_pos])
                body.set('pos', new_pos)

        # Update 'fromto' for geoms
        for i, name in enumerate(element_geom_names_first):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                index = i + 9
                new_fromto = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[index+4]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        # Update 'pos' for bodies
        for i, name in enumerate(element_geom_names_aux):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                index = i + 9
                # Assuming limb_lengths for bodies start after the last geom
                new_pos = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord in current_pos])
                body.set('pos', new_pos)


        for i, name in enumerate(element_geom_names_last):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                index = i + 17
                new_fromto = ' '.join([str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[index+4]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        # Save the modified XML file
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
        self.mat_limb_length = []
        self.mat_reward = []
        self.mat_iteration = []
        self.average_reward = []
        self.average_episode_length = []
        self.model_name = model_name
        self.mat_file_name = model_name
        self.design_iteration = [0 for _ in range(self.n_envs_train)]
        self.my_custom_condition = True  # Initialize your condition
        self.model.evaluate_current_policy = True
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/assets/"
        self.limb_radius_range = [0.01, 0.4]
        self.limb_length_range = [0.4, 2.0]
        self.torus_radius_range = [0.01, 0.4]

        self.limb_length = np.ones(25) * 0.5

    def _on_rollout_start(self) -> bool:

        # reset the environments
        for i in range(self.n_envs_train):

            self.limb_length = np.array([0.235454, 1.654685, 1.674746, 1.376238, 1.572604, 0.123518, 0.120762, 0.212702, 0.314148, 1.427817, 1.835761, 1.688915, 1.344858, 0.095647, 0.194717, 0.167639, 0.298142, 1.566769, 1.533329, 1.375999, 0.642733, 0.116480, 0.238949, 0.093317, 0.135114])


            self.modify_xml_ant_full_geometry(f"{self.mujoco_file_folder}ant_{i}.xml", self.limb_length)
            self.training_env.env_method('__init__', i, indices=[i])
            self.training_env.env_method("set_limb_length", self.limb_length, indices=[i])
            self.training_env.env_method('reset', indices=[i])
            # print(self.training_env.env_method('get_env_id', indices=[i]))

            # print(self.training_env.env_method('get_limb_length', indices=[i]))
        # time.sleep(20)

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
                    # Reset episode reward accumulator
                    self.episode_rewards[i] = 0
                    self.episode_length[i] = 0
        return True

    def _on_rollout_end(self) -> bool:

        for i in range(self.n_envs_train):
            current_limb_length = self.training_env.env_method('get_limb_length', indices=[i])[0]
            # Matlab logging
            self.mat_limb_length.append(current_limb_length)
            self.mat_reward.append(self.episode_rewards[i])
            self.mat_iteration.append(self.episode_length[i])
        self.logger.record("mean episode length", np.sum(self.average_episode_length) / np.sum(self.design_iteration))
        self.logger.record("mean reward", np.sum(self.average_reward) / np.sum(self.design_iteration))

        output_data = {
            "limb_length": np.array(self.mat_limb_length),
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_iteration),

        }
        print("saving matlab data...")
        file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/ant/rl/evaluation/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        self.average_episode_length = []
        self.average_reward = []
        self.design_iteration = [0 for _ in range(self.n_envs_train)]

        return True

    def modify_xml_geometry(self, file_path, limb_lengths):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - limb_lengths: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        limb_lengths = 0.7071 * limb_lengths
        # Names of the elements to modify
        element_body_names = ['front_left_foot', 'front_right_foot', 'back_foot', 'right_back_foot']
        element_geom_names_last = ['left_ankle_geom', 'right_ankle_geom', 'third_ankle_geom', 'fourth_ankle_geom']
        element_geom_names_first = ['left_leg_geom', 'right_leg_geom', 'back_leg_geom', 'rightback_leg_geom']

        # Update 'fromto' for geoms
        for i, name in enumerate(element_geom_names_first):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                new_fromto = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[i]) if float(coord) != 0 else '0' for coord in
                     current_fromto])
                geom.set('fromto', new_fromto)

        for i, name in enumerate(element_geom_names_last):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                index = i + len(element_geom_names_last)
                new_fromto = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord
                     in current_fromto])
                geom.set('fromto', new_fromto)

        # Update 'pos' for bodies
        for i, name in enumerate(element_body_names):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                # Assuming limb_lengths for bodies start after the last geom
                new_pos = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[i]) if float(coord) != 0 else '0' for coord in
                     current_pos])
                body.set('pos', new_pos)

        # Save the modified XML file
        tree.write(file_path)

    def modify_xml_ant_geometry(self, file_path, limb_lengths):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - limb_lengths: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        limb_lengths = 0.7071 * limb_lengths
        # Names of the elements to modify
        element_body_names = ['front_left_foot', 'front_right_foot', 'back_foot', 'right_back_foot']
        element_geom_names_last = ['left_ankle_geom', 'right_ankle_geom', 'third_ankle_geom', 'fourth_ankle_geom']
        element_geom_names_first = ['left_leg_geom', 'right_leg_geom', 'back_leg_geom', 'rightback_leg_geom']
        element_geom_thigh = ['aux_1_geom', 'aux_2_geom', 'aux_3_geom', 'aux_4_geom']
        element_body_thigh = ['aux_1', 'aux_2', 'aux_3', 'aux_4']

        # set new size for torso
        torso = root.findall(f".//geom[@name='torso_geom']")
        for geom in torso:
            new_size = ' '.join([str(float(limb_lengths[0]))])
            geom.set('size', new_size)

        for i, name in enumerate(element_geom_thigh):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                new_fromto = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[1]) if float(coord) != 0 else '0' for coord in
                     current_fromto])
                new_size = ' '.join([str(float(limb_lengths[2]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        for i, name in enumerate(element_body_thigh):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                # Assuming limb_lengths for bodies start after the last geom
                new_pos = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[1]) if float(coord) != 0 else '0' for coord in
                     current_pos])
                body.set('pos', new_pos)

        # Update 'fromto' for geoms
        for i, name in enumerate(element_geom_names_first):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                new_fromto = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[3]) if float(coord) != 0 else '0' for coord in
                     current_fromto])
                new_size = ' '.join([str(float(limb_lengths[4]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        # Update 'pos' for bodies
        for i, name in enumerate(element_body_names):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                # Assuming limb_lengths for bodies start after the last geom
                new_pos = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[3]) if float(coord) != 0 else '0' for coord in
                     current_pos])
                body.set('pos', new_pos)

        for i, name in enumerate(element_geom_names_last):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                new_fromto = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[5]) if float(coord) != 0 else '0' for coord in
                     current_fromto])
                new_size = ' '.join([str(float(limb_lengths[6]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        # Save the modified XML file
        tree.write(file_path)

    def modify_xml_ant_full_geometry(self, file_path, limb_lengths):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - limb_lengths: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        limb_lengths = 0.7071 * limb_lengths
        # Names of the elements to modify
        element_body_names = ['front_left_foot', 'front_right_foot', 'back_foot', 'right_back_foot']
        element_geom_names_last = ['left_ankle_geom', 'right_ankle_geom', 'third_ankle_geom', 'fourth_ankle_geom']
        element_geom_names_first = ['left_leg_geom', 'right_leg_geom', 'back_leg_geom', 'rightback_leg_geom']
        element_geom_thigh = ['aux_1_geom', 'aux_2_geom', 'aux_3_geom', 'aux_4_geom']
        element_body_thigh = ['aux_1', 'aux_2', 'aux_3', 'aux_4']
        element_geom_names_aux = ['left_leg_geom_aux', 'right_leg_geom_aux', 'back_leg_geom_aux',
                                  'rightback_leg_geom_aux']

        # set new size for torso
        torso = root.findall(f".//geom[@name='torso_geom']")
        for geom in torso:
            new_size = ' '.join([str(float(limb_lengths[0]))])
            geom.set('size', new_size)

        for i, name in enumerate(element_geom_thigh):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                index = i + 1
                new_fromto = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord
                     in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[index + 4]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        for i, name in enumerate(element_body_thigh):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                # Assuming limb_lengths for bodies start after the last geom
                index = i + 1
                new_pos = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord
                     in current_pos])
                body.set('pos', new_pos)

        # Update 'fromto' for geoms
        for i, name in enumerate(element_geom_names_first):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                index = i + 9
                new_fromto = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord
                     in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[index + 4]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        # Update 'pos' for bodies
        for i, name in enumerate(element_geom_names_aux):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:
                current_pos = body.get('pos').split(' ')
                index = i + 9
                # Assuming limb_lengths for bodies start after the last geom
                new_pos = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord
                     in current_pos])
                body.set('pos', new_pos)

        for i, name in enumerate(element_geom_names_last):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                current_fromto = geom.get('fromto').split(' ')
                index = i + 17
                new_fromto = ' '.join(
                    [str(float(coord) / abs(float(coord)) * limb_lengths[index]) if float(coord) != 0 else '0' for coord
                     in current_fromto])
                new_size = ' '.join([str(float(limb_lengths[index + 4]))])
                geom.set('size', new_size)
                geom.set('fromto', new_fromto)

        # Save the modified XML file
        tree.write(file_path)


if __name__ == '__main__':
    main()