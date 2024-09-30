import mujoco
import numpy as np
import glfw
import OpenGL.GL as gl
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from walker2d_v4 import Walker2dEnv
import xml.etree.ElementTree as ET
import random
import torch


ENV_ID = 17
mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/assets/"
limb_length = np.array([0.2, 0.228491, 0.129382, 0.1, 0.228491, 0.129382, 0.1])

#modify_xml_walker_full_geometry(f"{mujoco_file_folder}walker2d_{ENV_ID}.xml", limb_length)

# Paths
EVAL_MODEL_PATH = "/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/trained_model/random_design/walker_constant_1.zip"
XML_PATH = f"{mujoco_file_folder}walker2d.xml"


# Initialize GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(800, 600, "Ant Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

# Make the window's context current
glfw.make_context_current(window)
glfw.swap_interval(1)

# Load the model and data
xml_model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(xml_model)

# Initialize visualization structures
camera = mujoco.MjvCamera()
option = mujoco.MjvOption()
scene = mujoco.MjvScene(xml_model, maxgeom=10000)
context = mujoco.MjrContext(xml_model, mujoco.mjtFontScale.mjFONTSCALE_150)


# Main evaluation loop
batch_size = 8  # Define your batch size
episode_length_loop = 1024
n_envs_train = 1


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

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Define a custom environment creator function
def create_env():
    return Walker2dEnv(env_id=ENV_ID,render_mode='human')


def run_policy_with_rendering(eval_env, xml_model, model, data, camera, option, scene, context, steps):
    total_reward = 0
    obs = model.env.reset()

    for step in range(steps):
        # Get the action from the model's policy
        actions, _states  = model.predict(obs)

        # Apply the action to the simulation
        data.ctrl[:] = actions

        # Step the simulation forward
        mujoco.mj_step(xml_model, data)

        # Step the environment
        obs, rewards, dones, infos = model.env.step(actions)
        total_reward += rewards

        # Render the scene
        mujoco.mjv_updateScene(xml_model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), scene, context)
        glfw.swap_buffers(window)

        # Process GLFW events
        glfw.poll_events()

        if dones:
            break

    return total_reward


# Set global seed
global_seed = 1
set_random_seed(global_seed)


for iter in range(batch_size):

    eval_env = create_env()
    model = PPO.load(EVAL_MODEL_PATH, env=eval_env, seed=global_seed)
    total_reward = run_policy_with_rendering(eval_env, xml_model, model, data, camera, option, scene, context, episode_length_loop)
    print(f"Total reward for iteration {iter}: {total_reward}")

# Clean up and close the window
glfw.terminate()