import numpy as np
import random
from scipy.io import savemat
import xml.etree.ElementTree as ET


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

            for env_idx in range(self.n_envs_train):
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

                xml_file = f"{self.mujoco_file_folder}walker2d_{env_idx}.xml"
                self.modify_xml_walker_full_geometry(xml_file, self.limb_length)
                self.training_env.env_method('reset', indices=[env_idx])

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
                        self.env.env_method('reset', indices=[i])

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
