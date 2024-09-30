from stable_baselines3 import PPO
from ant_v4 import AntEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from scipy.io import savemat
import time
import math as m
import xml.etree.ElementTree as ET
import os
import shutil

ENV_ID = 5

def main():

    # original_xml_path = f"/home/divij/master_thesis/src/ant/assets/ant.xml"
    # destination_folder = f"/home/divij/master_thesis/src/ant/assets/"
    # # Copy the original file with new names
    # for i in range(10):
    #     new_file_name = f"ant_{i}.xml"
    #     new_file_path = os.path.join(destination_folder, new_file_name)
    #     shutil.copy2(original_xml_path, new_file_path)

    episode_length_loop = 1000
    batch_size = 10

    # Training parameters
    while True:
        # limb_length = np.array([0.231805188384530,	1.89678229967314,	1.89969515581709,	0.962609123914754,
        #                         1.89915859699249,	0.120651330143616,	0.157418370246887,	0.220062181353569,
        #                         0.145438432693481,	1.88376514614312,	1.46022522449493,	1.77137747071596,
        #                         1.62732862164700,	0.0487723672471488,	0.0983623936772347,	0.142752199766444,
        #                         0.226030036807060,	1.20895024453829,	1.28940320014954,	1.96461844444275,
        #                         0.618850998206947,	0.0815493762493134,	0.225574567913945,	0.0442982465038912,
        #                         0.185997530817986])
        # limb_length = np.array([0.242420952071230,	1.84207541906573,	1.93100387599680,	1.11417774105968,
        #                         1.86779265141941,	0.128002304902217,	0.111981322011874,	0.223440748724831,
        #                         0.145718997685779,	1.86143350707532,	1.45671847976557,	1.81286217768863,
        #                         1.62695204376338,	0.0474446765448892,	0.146598986099355,	0.143883110822275,
        #                         0.213704838691681,	1.16683281707246,	1.28465833531145,	1.93533621690022,
        #                         0.599258766684416,	0.0804422149207296,	0.212034440757231,	0.0473346705309564,
        #                         0.193739036162652])
        limb_length = np.array([0.245810490831137,	1.74792355231554,	1.90190336565184,	1.13141192142827,
                                1.78321946284236,	0.220869707533156,	0.228147382257129,	0.256958519635907,
                                0.223184123541184,	1.33385360131245,	1.31986140381957,	1.41907215936511,
                                1.56334179957695,	0.121354605548262,	0.110907315647922,	0.162318922886031,
                                0.275619002671292,	1.09589781277966,	1.03199159029180,	1.77575356136284,
                                0.686110354322779,	0.149259002377896,	0.238485756569595,	0.0967745723436930,
                                0.208594050700340])
        mujoco_file_folder = f"/home/divij/master_thesis/src/ant/assets/"
        EVAL_MODEL_PATH = f"/home/divij/master_thesis/src/ant/rl/trained_model/allbody/bestGaussDesign_ant_FullGeom_Hebo_Gauss_callback_envs_50_EpLen_1024_lr_0.0001_hid_size_256_FORreward_1.0_CTRLcost_0.0"
        modify_xml_ant_full_geometry(f"{mujoco_file_folder}ant_{ENV_ID}.xml", limb_length)

        for iter in range(batch_size):
            eval_env = create_env()            
            model = PPO.load(EVAL_MODEL_PATH, env=eval_env)

            rewards = 0
            episode_length = 0
            obs = model.env.reset()

            for _ in range(episode_length_loop):
                actions, _states = model.predict(obs)
                obs, rewards_env, dones, infos = model.env.step(actions)
                rewards += rewards_env 
                episode_length += 1
                # print("Reward: ", rewards_env)
                time.sleep(0.05)
                # if dones:
                #     break            

def create_env():
    return AntEnv(env_id=ENV_ID,render_mode='human')

def modify_xml_geometry(file_path, limb_lengths):
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

def modify_xml_ant_geometry(file_path, limb_lengths):
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

def modify_xml_ant_full_geometry(file_path, limb_lengths):
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
    for i, name in enumerate(element_body_names):
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

if __name__ == '__main__':
    main()
