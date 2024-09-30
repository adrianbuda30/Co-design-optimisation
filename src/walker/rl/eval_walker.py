from stable_baselines3 import PPO
from walker2d_v4 import Walker2dEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from scipy.io import savemat
import time
import math as m
import xml.etree.ElementTree as ET
import os
import shutil
import mujoco
import gym

ENV_ID = 26

def main():

    episode_length_loop = 1024
    batch_size = 10

    limb_length = np.array([0.2, 0.20, 0.27, 0.1, 0.20, 0.27, 0.1])
    mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/assets/"
    EVAL_MODEL_PATH = "/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/trained_model/random_design/walker_test_1.zip"
    # modify_xml_walker_full_geometry(f"{mujoco_file_folder}walker2d_{ENV_ID}.xml", limb_length)

    XML_PATH = f"{mujoco_file_folder}walker2d.xml"

    # Load the model and data
    xml_model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(xml_model)

    for iter in range(batch_size):
        eval_env = gym.make('Walker2d-v4')
        model = PPO.load(EVAL_MODEL_PATH, env=eval_env)

        rewards = 0
        episode_length = 0
        obs = model.env.reset()

        for _ in range(episode_length_loop):
            actions, _states = model.predict(obs, deterministic=True)
            data.ctrl[:] = actions
            mujoco.mj_step(xml_model, data)

            obs, rewards_env, dones, infos = model.env.step(actions)
            rewards += rewards_env
            episode_length += 1
            if dones:
                 break

        print("Reward: ", rewards)

def create_env():
    return Walker2dEnv(env_id=ENV_ID,render_mode='human')

def modify_xml_walker_full_geometry(file_path, limb_lengths):
    """
    Modify 'fromto' attributes for specified geoms and 'pos' attributes for specified bodies in an XML file based on new limb lengths while maintaining the original sign.

    Args:
    - file_path: Path to the XML file to modify.
    - limb_lengths: Sequence containing the new limb lengths, maintaining the sign.
    """
    # Load the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()


    torso_length = limb_lengths[0]
    thigh_length = limb_lengths[1]
    leg_length = limb_lengths[2]
    foot_length = limb_lengths[3]

    # Names of the elements to modify
    element_body_names = ['thigh', 'leg', 'foot', 'thigh_left', 'leg_left', 'foot_left']
    element_geom_names = ['thigh_geom', 'leg_geom', 'foot_geom', 'thigh_left_geom', 'leg_left_geom',
                          'foot_left_geom']

    # Set new size for torso (if needed)
    torso_geom = root.findall(".//geom[@name='torso_geom']")
    for geom in torso_geom:
        current_size = geom.get('size').split(' ')
        new_size = current_size[0:1] + [str(limb_lengths[0])]  # Change only the first element of the size
        geom.set('size', ' '.join(new_size))

    torso = root.findall(".//body[@name='torso']")
    for body in torso:
        current_pos = body.get('pos').split(' ')
        new_pos = current_pos[0:2] + [str(0.10000000000000001 + 2 * leg_length + torso_length + 2 * thigh_length)]
        body.set('pos', ' '.join(new_pos))

    # Set new size and position for legs and other parts
    for i, name in enumerate(element_geom_names):
        geoms = root.findall(f".//geom[@name='{name}']")
        for geom in geoms:
            index = i + 1
            current_size = geom.get('size').split(' ')
            new_size = current_size[0:1] + [str(limb_lengths[index])]
            geom.set('size', ' '.join(new_size))

            if 'pos' in geom.attrib:
                if 'thigh' in name:
                    new_geom_pos = [0, 0, -thigh_length]
                elif 'foot' in name:
                    new_geom_pos = [-foot_length, 0, 0.10000000000000001]

                # Update the position
                geom.set('pos', ' '.join(map(str, new_geom_pos)))

    for i, name in enumerate(element_body_names):
        bodies = root.findall(f".//body[@name='{name}']")
        for body in bodies:
            # Calculate new position based on the lengths of the preceding body parts
            if 'thigh' in name:
                new_body_pos = [0, 0, - torso_length]
            elif 'leg' in name:
                new_body_pos = [0, 0, - 2 * thigh_length - leg_length]
            elif 'foot' in name:
                new_body_pos = [2 * foot_length, 0, - leg_length - 0.10000000000000001]

            # Update the position
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
    # Save the modified XML file
    tree.write(file_path)


if __name__ == '__main__':
    main()
