
import os
import shutil
from sbx import PPO
import torch
from jax import nn
import numpy as np
import random
from scipy.io import savemat
import xml.etree.ElementTree as ET


from walker2d_v4 import Walker2dEnv

def main():
    # Training parameters
    use_sde = False
    hidden_sizes_train = 256
    learning_rate_train = 0.0001
    n_epochs_train = 10
    LOAD_OLD_MODEL = False
    n_steps_train = 512 * 2
    n_envs_train = 1
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 10000
    batch_size_train = 128
    global_iteration = 0
    model_name = "walker_random_trial_sbx"

    # Paths for original and destination XML files
    original_xml_path = "/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/assets/walker2d.xml"
    destination_folder = "/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/assets/"

    # Copy the XML file to create different environment files for each parallel environment
    for i in range(n_envs_train):
        new_file_name = f"walker2d_{i}.xml"
        new_file_path = os.path.join(destination_folder, new_file_name)
        shutil.copy2(original_xml_path, new_file_path)

    # Set up environment
    env_config = [{'env_id': i, 'ctrl_cost_weight': 0.5} for i in range(n_envs_train)]
    single_env_config = env_config[0]
    env = Walker2dEnv(**single_env_config)
    log_dir = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/walker_tensorboard/TB_{model_name}"

    # Define model architecture
    onpolicy_kwargs = dict(
        activation_fn=nn.tanh,
        net_arch=dict(vf=[hidden_sizes_train, hidden_sizes_train],
                      pi=[hidden_sizes_train, hidden_sizes_train])
    )

    # Load or initialize the PPO model
    if LOAD_OLD_MODEL:
        old_model_path = "/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/rl/trained_model/bestDesign_walker_Schaff_8param_nobuckling_1distrib"
        old_model = PPO.load(old_model_path, env=env)
        new_model = PPO("MlpPolicy", env=env, n_steps=n_steps_train, batch_size=batch_size_train,
                        n_epochs=n_epochs_train, use_sde=use_sde, ent_coef=entropy_coeff_train,
                        learning_rate=learning_rate_train, policy_kwargs=onpolicy_kwargs,
                        device='cpu', verbose=1, tensorboard_log=log_dir)
        new_model.set_parameters(old_model.get_parameters())
    else:
        new_model = PPO("MlpPolicy", env=env, n_steps=n_steps_train, batch_size=batch_size_train,
                        n_epochs=n_epochs_train, use_sde=use_sde, ent_coef=entropy_coeff_train,
                        learning_rate=learning_rate_train, policy_kwargs=onpolicy_kwargs,
                        device='cpu', verbose=1, tensorboard_log=log_dir)
        print("New model created")

    print("Model training...")

    # Create an instance of RandomDesignTrainer
    trainer = RandomDesignTrainer(
        model=new_model,
        env=env,
        model_name=model_name,
        n_steps_train=n_steps_train,
        n_envs_train=n_envs_train
    )

    # Train model using RandomDesignTrainer
    trainer.train(total_timesteps=total_timesteps_train)

    print("Model trained, saving...")
    save_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/trained_model/random_design/{model_name}"
    new_model.save(save_path)
    print(f"Model saved at {save_path}")

    env.close()
    print("Training complete.")


class RandomDesignTrainer:
    def __init__(self, model, env, model_name="matfile", n_steps_train=5120, n_envs_train=1):
        self.model = model
        self.env = env
        self.n_steps_train = n_steps_train
        self.n_envs_train = n_envs_train
        self.episode_rewards = [0] * n_envs_train
        self.episode_length = [0] * n_envs_train
        self.design_iteration = [0] * n_envs_train
        self.mat_limb_length = []
        self.mat_reward = []
        self.mat_iteration = []
        self.average_reward = []
        self.average_episode_length = []
        self.model_name = model_name
        self.limb_length_range = [0.1, 1.0]
        self.foot_length_range = [0.1, 0.4]
        self.limb_thickness_range = [0.01, 0.05]
        self.mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/assets/"
        self.reset_limb_length()

    def reset_limb_length(self):
        # Randomize limb lengths at the start of each rollout
        self.limb_length = np.array([
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1] / 2),
                                        # torso
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),  # thigh
                                        random.uniform(self.limb_length_range[0], self.limb_length_range[1]),  # shin
                                        random.uniform(self.foot_length_range[0], self.foot_length_range[1])  # foot
                                    ] * 2)  # repeat for left/right limbs

    def modify_xml_walker_full_geometry(self, file_path, limb_lengths):
        """
        Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

        Args:
        - file_path: Path to the XML file to modify.
        - limb_lengths: Sequence containing the new limb lengths, maintaining the sign.
        """
        # Load the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        original_lengths = {
            'torso': 0.2,  # Default length of torso
            'thigh': 0.225,  # Default length of thigh
            'leg': 0.25,  # Default length of leg
            'foot': 0.1,  # Default length of foot
            'torso_thick': 0.05,  # Default length of torso
            'thigh_thick': 0.05,  # Default length of thigh
            'leg_thick': 0.04,  # Default length of leg
            'foot_thick': 0.06  # Default length of foot
        }

        torso_length = limb_lengths[0]
        thigh_length = limb_lengths[1]
        leg_length = limb_lengths[2]
        foot_length = limb_lengths[3]

        torso_thickness = limb_lengths[7]
        thigh_thickness = limb_lengths[8]
        leg_thickness = limb_lengths[9]
        foot_thickness = limb_lengths[10]


        element_body_names = ['thigh', 'leg', 'foot', 'thigh_left', 'leg_left', 'foot_left']
        element_geom_names = ['thigh_geom', 'leg_geom', 'foot_geom', 'thigh_left_geom', 'leg_left_geom',
                              'foot_left_geom']


        torso_geom = root.findall(".//geom[@name='torso_geom']")
        for geom in torso_geom:
            current_size = geom.get('size').split(' ')
            new_size = [str(limb_lengths[7])] + [str(limb_lengths[0])]
            geom.set('size', ' '.join(new_size))

        torso = root.findall(".//body[@name='torso']")
        for body in torso:
            current_pos = body.get('pos').split(' ')
            new_pos = current_pos[0:2] + [str(0.10000000000000001 + 2 * leg_length + torso_length + 2 * thigh_length)]
            body.set('pos', ' '.join(new_pos))


        for i, name in enumerate(element_geom_names):
            geoms = root.findall(f".//geom[@name='{name}']")
            for geom in geoms:
                index = i + 1
                current_size = geom.get('size').split(' ')
                new_size = [str(limb_lengths[index + 7])] + [str(limb_lengths[index])]
                geom.set('size', ' '.join(new_size))

                if 'pos' in geom.attrib:
                    if 'thigh' in name:
                        new_geom_pos = [0, 0, -thigh_length]
                    elif 'foot' in name:
                        new_geom_pos = [-foot_length, 0, 0.10000000000000001]


                    geom.set('pos', ' '.join(map(str, new_geom_pos)))

        for i, name in enumerate(element_body_names):
            bodies = root.findall(f".//body[@name='{name}']")
            for body in bodies:

                if 'thigh' in name:
                    new_body_pos = [0, 0, - torso_length]
                elif 'leg' in name:
                    new_body_pos = [0, 0, - 2 * thigh_length - leg_length]
                elif 'foot' in name:
                    new_body_pos = [2 * foot_length, 0, - leg_length - 0.10000000000000001]


                body.set('pos', ' '.join(map(str, new_body_pos)))

            joints = root.findall(f".//joint[@name='{name}_joint']")
            for joint in joints:
                if 'pos' in joint.attrib:
                    if 'thigh' in name:
                        joint_pos = [0, 0, 0]
                    elif 'leg' in name:
                        joint_pos = [0, 0, leg_length]
                    elif 'foot' in name:
                        joint_pos = [-2 * foot_length, 0, 0.10000000000000001]
                    joint.set('pos', ' '.join(map(str, joint_pos)))

        tree.write(file_path)

    def train(self, total_timesteps):
        timesteps = 0
        while timesteps < total_timesteps:

            for i in range(self.n_envs_train):
                # Update geometry for each environment
                self.torso = random.uniform(self.limb_length_range[0], self.limb_length_range[1] / 2)
                self.thigh = random.uniform(self.limb_length_range[0], self.limb_length_range[1])
                self.shin = random.uniform(self.limb_length_range[0], self.limb_length_range[1])
                self.foot = random.uniform(self.foot_length_range[0], self.foot_length_range[1])
                self.thickness_torso = random.uniform(self.limb_thickness_range[0], self.limb_thickness_range[1])
                self.thickness_thigh = random.uniform(self.limb_thickness_range[0], self.limb_thickness_range[1])
                self.thickness_shin = random.uniform(self.limb_thickness_range[0], self.limb_thickness_range[1])
                self.thickness_foot = random.uniform(self.limb_thickness_range[0], self.limb_thickness_range[1])
                self.limb_length = np.array(
                    [self.torso, self.thigh, self.shin, self.foot, self.thigh, self.shin, self.foot,
                     self.thickness_torso, self.thickness_thigh, self.thickness_shin, self.thickness_foot,
                     self.thickness_thigh, self.thickness_shin, self.thickness_foot])

                xml_file = f"{self.mujoco_file_folder}walker2d_{i}.xml"
                self.env.__init__()
                self.modify_xml_walker_full_geometry(xml_file, self.limb_length)
                self.env.set_limb_length(self.limb_length)
                self.env.reset_model()

            # Run the training loop
            for step in range(self.n_steps_train):
                actions = self.model.predict(self.env)
                observations, rewards, dones, _ = self.env.step(actions)

                # Collect rewards and update lengths
                for i in range(self.n_envs_train):
                    self.episode_rewards[i] += rewards[i]
                    self.episode_length[i] += 1
                    if dones[i] or self.episode_length[i] >= self.n_steps_train:
                        self.average_reward.append(self.episode_rewards[i])
                        self.average_episode_length.append(self.episode_length[i])
                        self.design_iteration[i] += 1
                        self.episode_rewards[i] = 0
                        self.episode_length[i] = 0
                        # Reset environment
                        self.env.reset()

                timesteps += self.n_envs_train
                if timesteps >= total_timesteps:
                    break

            # Save data after each rollout
            self.save_data()

    def save_data(self):
        # Save collected data to .mat file
        output_data = {
            "limb_length": np.array(self.mat_limb_length),
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_iteration),
        }
        file_path = f"/path/to/your/trained_model/{self.model_name}.mat"
        savemat(file_path, output_data)
        print(f"Data saved to {file_path}")


if __name__ == "__main__":
    main()