import gym
import torch
import rospy
import tf.transformations as tf_trans
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sb_quadcopter_env import QuadcopterEnv
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat, savemat
import geometry_msgs.msg

# Constants
ON_POLICY_KWARGS = dict(activation_fn=torch.nn.Tanh, net_arch=dict(vf=[128, 128], pi=[128, 128]))
INITIAL_POS = np.array([0.0, 0.0, 0.0])
TARGET_POS = np.array([0.0, 0.0, 0.0])
LEARNING_RATE = 0.003
N_EPOCHS = 10
N_STEPS = 5000
N_ENVS = 8
EVAL_COUNTER = 0
TOTAL_TIMESTEPS = N_STEPS * N_ENVS * 1
EVAL_MODEL_PATH = "/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/Quadcopter_Hebo_callback_Tanh_Tsteps_122880000_lr_0.0001_hidden_sizes_256_POSreward_0.5_VELreward_0.5_omega_pen_0.1_preOpt"
LOOP_COUNT = 100000


def rotation_matrix_to_quaternion(rotation_matrix: np.array) -> tuple:
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


def make_marker(id: int, color: tuple, frame_id: str = "world", ns: str = "quadcopter") -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = id
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r, marker.color.g, marker.color.b = color
    return marker

def make_marker_traj(id: int, color: tuple, frame_id: str = "world", ns: str = "quadcopter", marker_type=Marker.SPHERE) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = id
    marker.type = marker_type
    marker.action = marker.ADD
    marker.scale.x = 0.005  # Thickness of the LINE_STRIP
    if marker_type == Marker.SPHERE:
        marker.scale.y = marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r, marker.color.g, marker.color.b = color
    return marker

def update_and_publish_marker(marker: Marker, pose: dict, publisher: rospy.Publisher) -> None:
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = pose.values()
    publisher.publish(marker)
 

def main():
    rospy.init_node('frames_publisher_node')
    br = TransformBroadcaster()
    marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    marker_init = make_marker(0, (1.0, 0.0, 0.0))
    marker_target = make_marker(1, (0.0, 1.0, 0.0))

    mesh_marker = Marker()
    # Initialize the marker and publisher outside the loop
    mesh_marker.header.frame_id = "world"
    mesh_marker.ns = "quadcopter_mesh"
    mesh_marker.id = 0
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.action = Marker.ADD
    mesh_marker.scale.x = mesh_marker.scale.y = mesh_marker.scale.z = 1.0  # adjust these values as needed
    mesh_marker.color.a = 1.0
    mesh_marker.color.r = mesh_marker.color.g = mesh_marker.color.b = 1.0  # adjust these for the desired color
    mesh_marker.mesh_resource = "package://model_dynamics/mesh/quadrotor.dae"  # path to the .dae file
    mesh_publisher = rospy.Publisher('visualization_marker_mesh', Marker, queue_size=10)

    trajectory_init = make_marker_traj(2, (1.0, 0.0, 0.0), marker_type=Marker.LINE_STRIP)
    trajectory_target = make_marker_traj(3, (0.0, 1.0, 0.0), marker_type=Marker.LINE_STRIP)
    trajectory_quadcopter = make_marker_traj(4, (0.0, 0.0, 1.0), marker_type=Marker.LINE_STRIP)  # Blue for quadcopter

    obs, sensor_motor_rpm, action_motor_rpm, omega_model, velocity_model, position_model, omega, R = [], [], [], [], [], [], [], []


    while True:
        global EVAL_COUNTER
        EVAL_COUNTER += 1
        print("evaluation starts at 5: ", EVAL_COUNTER)
        if EVAL_COUNTER == 1:
            EVAL_COUNTER = 0
            print("Model trained, evaluating...")
            eval_env = make_vec_env(QuadcopterEnv, n_envs=1)
            eval_model = PPO.load(EVAL_MODEL_PATH, env=eval_env)
            obs = eval_env.reset()
            dones = False
            for i in range(LOOP_COUNT):
                if dones:
                    obs = eval_env.reset()
                    print(obs)
                action, _states = eval_model.predict(obs)
                obs, rewards, dones, info = eval_env.step(action)
                
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "world"
                t.child_frame_id = "av1"
                t.transform.translation.x = info[0]['pos_world'][0]
                t.transform.translation.y = info[0]['pos_world'][1]
                t.transform.translation.z = info[0]['pos_world'][2]
                q = rotation_matrix_to_quaternion(np.reshape(info[0]['rot_matrix_world'], (3, 3)))
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]

                # Publisher for the MeshResource marker
                mesh_marker.pose.position.x = info[0]['pos_world'][0]
                mesh_marker.pose.position.y = info[0]['pos_world'][1]
                mesh_marker.pose.position.z = info[0]['pos_world'][2]
                mesh_marker.pose.orientation.x = q[0]
                mesh_marker.pose.orientation.y = q[1]
                mesh_marker.pose.orientation.z = q[2]
                mesh_marker.pose.orientation.w = q[3]


                position_model.append(np.array(info[0]['pos_world']))        
                velocity_model.append(np.array(info[0]['vel_world']))
                omega_model.append(np.array(info[0]['omega_world']))
                R.append(info[0]['rot_matrix_world'])  # Reshape into 3x3 matrix before appending
                action_motor_rpm.append(np.array(info[0]['action_motor_rpm']))
                sensor_motor_rpm.append(np.array(info[0]['propeller_speed']))
                # print(info[0]['propeller_speed'])
                mesh_publisher.publish(mesh_marker)  # Publish the mesh marker
                # print(f"reward: {info[0]['reward']:<7.2f} distance: {info[0]['distance']:<7.2f} action: {info[0]['action'][0]:<7.2f} {info[0]['action'][1]:<7.2f} {info[0]['action'][2]:<7.2f} {info[0]['action'][3]:<7.2f} propeller_speed: {info[0]['propeller_speed'][0]:<7.2f} {info[0]['propeller_speed'][1]:<7.2f} {info[0]['propeller_speed'][2]:<7.2f} {info[0]['propeller_speed'][3]:<7.2f} ")
                # print(f"reward: {info[0]['reward']:<7.2f} action: {info[0]['action'][0]:<7.2f} {info[0]['action'][1]:<7.2f} {info[0]['action'][2]:<7.2f} {info[0]['action'][3]:<7.2f} omega: {info[0]['vel_world'][0]:<7.2f} {info[0]['vel_world'][1]:<7.2f} {info[0]['vel_world'][2]:<7.2f}, speed {info[0]['propeller_speed'][0]:<7.2f} {info[0]['propeller_speed'][1]:<7.2f} {info[0]['propeller_speed'][2]:<7.2f} {info[0]['propeller_speed'][3]:<7.2f}")

                # print(f"reward: {info[0]['reward']:<7.2f} distance: {info[0]['distance']:<7.2f} target_reward_norm: {info[0]['target_reward_norm']:<7.2f} action_penalty: {info[0]['action_penalty']:<7.2f} omega_world_penalty: {info[0]['omega_world_penalty']:<7.2f}  action: {info[0]['action'][0]:<7.2f} {info[0]['action'][1]:<7.2f} {info[0]['action'][2]:<7.2f} {info[0]['action'][3]:<7.2f} ")
                # print(f"reward: {info[0]['reward']:<7.2f} distance: {info[0]['distance']:<7.2f} R: {info[0]['rot_matrix_world'][0]:<7.2f} {info[0]['rot_matrix_world'][1]:<7.2f} {info[0]['rot_matrix_world'][2]:<7.2f} {info[0]['rot_matrix_world'][3]:<7.2f} {info[0]['rot_matrix_world'][4]:<7.2f} {info[0]['rot_matrix_world'][5]:<7.2f} {info[0]['rot_matrix_world'][6]:<7.2f} {info[0]['rot_matrix_world'][7]:<7.2f} {info[0]['rot_matrix_world'][8]:<7.2f} quat: {q[0]:<7.2f} {q[1]:<7.2f} {q[2]:<7.2f} {q[3]:<7.2f}  action: {info[0]['action'][0]:<7.2f} {info[0]['action'][1]:<7.2f} {info[0]['action'][2]:<7.2f} {info[0]['action'][3]:<7.2f}")
                velocity_error = info[0]['vel_world'] - info[0]['desired_velocity']
                position_error = info[0]['pos_world'] - info[0]['target_pos']

                print(f"reward: {info[0]['reward']:<7.2f}, pos_error: {position_error[0]:<7.2f} {position_error[1]:<7.2f} {position_error[2]:<7.2f}, vel_error: {velocity_error[0]:<7.2f} {velocity_error[1]:<7.2f} {velocity_error[2]:<7.2f}, vel_world: {info[0]['vel_world'][0]:<7.2f} {info[0]['vel_world'][1]:<7.2f} {info[0]['vel_world'][2]:<7.2f}, vel_des: {info[0]['desired_velocity'][0]:<7.2f} {info[0]['desired_velocity'][1]:<7.2f} {info[0]['desired_velocity'][2]:<7.2f}")
                
                point_init = geometry_msgs.msg.Point()
                point_init.x, point_init.y, point_init.z = info[0]['init_pos']
                trajectory_init.points.append(point_init)

                point_quadcopter = geometry_msgs.msg.Point()
                point_quadcopter.x, point_quadcopter.y, point_quadcopter.z = info[0]['pos_world']
                trajectory_quadcopter.points.append(point_quadcopter)

                point_target = geometry_msgs.msg.Point()
                point_target.x, point_target.y, point_target.z = info[0]['target_pos']
                trajectory_target.points.append(point_target)

                marker_publisher.publish(trajectory_init)
                marker_publisher.publish(trajectory_target)
                marker_publisher.publish(trajectory_quadcopter)

                update_and_publish_marker(marker_init, {'x': info[0]['init_pos'][0], 'y': info[0]['init_pos'][1], 'z': info[0]['init_pos'][2]}, marker_publisher)
                update_and_publish_marker(marker_target, {'x': info[0]['target_pos'][0], 'y': info[0]['target_pos'][1], 'z': info[0]['target_pos'][2]}, marker_publisher)
                br.sendTransform(t)


        #save data in a matlab file
        output_data = {
            "position_model": np.array(position_model),
            "rotation_matrix_model": np.array(R),
            "action_motor_rpm": np.array(action_motor_rpm),
            "sensor_motor_rpm": np.array(sensor_motor_rpm),
            "velocity_model": np.array(velocity_model),
            "omega_model": np.array(omega_model),
        }
        print("saving to output.mat...")
        savemat('/home/divij/Desktop/orig_model/output_wind_10.mat', output_data)
        break


if __name__ == '__main__':
    main()
