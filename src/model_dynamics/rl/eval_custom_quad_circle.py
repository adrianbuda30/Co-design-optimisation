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
from env_quad_traj_circle import QuadcopterEnv
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat, savemat
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion, Point
import tf
from stable_baselines3.common.vec_env import SubprocVecEnv


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
EVAL_MODEL_PATH = "/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/random_design/QuadcopterCircle_length_random_design_start_upd_25lr_mean0.01_std_0.01_sdeFalse_Tanh_Tsteps_256000000_lr_0.0001_hidden_sizes_256_POSreward_0.0_VELreward_1.0_omega_pen_0.2_var_design"
LOOP_COUNT = 100000



# Function to publish a coordinate system using markers
def publish_quadcopter_coordinate_system(marker_pub, frame_id, position, orientation, scale=1.0):
    """
    Publishes a coordinate frame using markers to represent the position and orientation of a quadcopter center in RViz.

    Args:
    - marker_pub: The ROS publisher object for the Marker message.
    - frame_id: The frame ID in which the markers should be placed.
    - position: A tuple (x, y, z) representing the position of the quadcopter center.
    - orientation: A Quaternion representing the orientation of the quadcopter.
    - scale: A float to scale the size of the coordinate axes.
    """
    # Normalize the quaternion to use it for the rotation matrix
    quat_array = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
    quat_array /= np.linalg.norm(quat_array)
    orientation_normalized = Quaternion(*quat_array)

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_matrix(orientation_normalized)

    # Define axes in local frame
    axes = [np.array([scale, 0, 0]), np.array([0, scale, 0]), np.array([0, 0, scale])]
    colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]  # RGB colors

    for i, (axis, color) in enumerate(zip(axes, colors)):
        # Define a marker for each axis
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = i

        # Set the scale of the marker (width, height, arrowhead)
        marker.scale.x = 0.02 * scale  # Shaft diameter
        marker.scale.y = 0.04 * scale  # Head diameter
        marker.scale.z = 0.1 * scale   # Head length

        # Set the color and transparency
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color


        # Convert the direction to the global frame using the rotation matrix
        global_direction = np.dot(rotation_matrix[:3, :3], axis)

        # Set the points for the arrow
        start_point = geometry_msgs.msg.Point(*position)
        end_point = geometry_msgs.msg.Point(position[0] + global_direction[0],
                                            position[1] + global_direction[1],
                                            position[2] + global_direction[2])

        marker.points.append(start_point)
        marker.points.append(end_point)

        # Publish the marker
        marker_pub.publish(marker)


# Function to create a quaternion from a direction vector
def vector_to_quaternion(direction):
    # Create a quaternion that rotates the z-axis to align with the direction vector
    axis = np.cross([0, 0, 1], direction)
    angle = np.arccos(np.dot([0, 0, 1], direction))
    return Quaternion(*tf.transformations.quaternion_about_axis(angle, axis))

def quaternion_to_matrix(quat):
    return tf.transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])

def matrix_to_quaternion(matrix):
    q = tf.transformations.quaternion_from_matrix(matrix)
    return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

def publish_quadcopter_arms(marker_pub, frame_id, arm_lengths, position, global_orientation):
    # Ensure position is a tuple of floats
    position = tuple(float(p) for p in position)

    # Normalize the quaternion to use it for the rotation matrix
    quat_array = np.array([global_orientation.x, global_orientation.y, global_orientation.z, global_orientation.w])
    quat_array /= np.linalg.norm(quat_array)
    global_orientation_normalized = Quaternion(x=float(quat_array[0]), y=float(quat_array[1]), z=float(quat_array[2]), w=float(quat_array[3]))

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_matrix(global_orientation_normalized)

    # Define the directions for each arm in the quadcopter's body frame
    arm_directions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, -1, 0])]

    # Publish markers for each arm
    for i, direction in enumerate(arm_directions):
        # Convert the direction to the global frame using the rotation matrix
        global_direction = rotation_matrix[:3, :3].dot(direction)

        # Create the quaternion for the marker orientation
        marker_orientation = Quaternion(
            *tf.transformations.quaternion_about_axis(
                np.arccos(np.dot([0, 0, 1], global_direction)), 
                np.cross([0, 0, 1], global_direction) if not np.allclose(direction, [0, 0, 1]) else [0, 1, 0]
            )
        )

        # Create the marker for the arm
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Cylinder diameter
        marker.scale.y = 0.05  # Cylinder diameter
        marker.scale.z = float(arm_lengths[i])  # Use the specific arm length
        marker.color.a = 1.0  # Alpha
        #different color for each arm
        if i == 0:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        
        elif i == 1:    
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        
        elif i == 2:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0

        elif i == 3:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0

        # Set the position of the marker
        marker.pose.position = Point(
            x=position[0] + global_direction[0] * arm_lengths[i] / 2,
            y=position[1] + global_direction[1] * arm_lengths[i] / 2,
            z=position[2] + global_direction[2] * arm_lengths[i] / 2
        )

        # Set the orientation of the marker
        marker.pose.orientation = marker_orientation

        # Set a unique ID for each marker
        marker.id = i

        # Publish the marker
        marker_pub.publish(marker)



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


def main():
    rospy.init_node('frames_publisher_node')
    br = TransformBroadcaster()
    marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    marker_publisher_coord = rospy.Publisher('visualization_marker_coord', Marker, queue_size=10)
    obs, sensor_motor_rpm, action_motor_rpm, omega_model, velocity_model, position_model, omega, R = [], [], [], [], [], [], [], []
    prop_speeds_history = [[] for _ in range(4)]  # Create a list of empty lists to store the propeller speeds

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

    while True:
        global EVAL_COUNTER
        EVAL_COUNTER += 1
        print("evaluation starts at 5: ", EVAL_COUNTER)
        if EVAL_COUNTER == 1:
            EVAL_COUNTER = 0
            print("Model trained, evaluating...")
            REWARD = np.array([0.0, 1.0, 0.2])
            arm_length = np.array([0.424358, 0.319840, 0.103578, 0.254953])

            # Define unique initialization variables for each environment
            env_configs = [{'REWARD': REWARD, 'env_id': i, 'arm_length': arm_length} for i in range(1)]

            # Create function for each environment instance with its unique configuration
            env_fns = [lambda config=config: QuadcopterEnv(**config) for config in env_configs]

            # Create the vectorized environment using SubprocVecEnv directly
            eval_env = SubprocVecEnv(env_fns, start_method='fork')
            eval_model = PPO.load(EVAL_MODEL_PATH, env=eval_env)
            obs = eval_env.reset()
            dones = False
            for i in range(LOOP_COUNT):
                if dones:
                    obs = eval_env.reset()
                    print(obs)
                action, _states = eval_model.predict(obs)
                obs, rewards, dones, info = eval_env.step(action)
                
                position_model.append(np.array(info[0]['pos_world']))        
                velocity_model.append(np.array(info[0]['vel_world']))
                omega_model.append(np.array(info[0]['omega_world']))
                R.append(info[0]['rot_matrix_world'])  # Reshape into 3x3 matrix before appending
                action_motor_rpm.append(np.array(info[0]['action_motor_rpm']))
                sensor_motor_rpm.append(np.array(info[0]['propeller_speed']))


                rotation_matrix = np.reshape(info[0]['rot_matrix_world'], (3, 3))
                q = rotation_matrix_to_quaternion(rotation_matrix)

                # mesh_publisher.publish(mesh_marker)  # Publish the mesh marker
                # Now use this information to publish the quadcopter arms
                publish_quadcopter_arms(
                    marker_pub=marker_publisher,
                    frame_id="world",
                    arm_lengths=info[0]['arm_length'],  # Example arm length
                    position=(info[0]['pos_world'][0], info[0]['pos_world'][1], info[0]['pos_world'][2]),
                    global_orientation=Quaternion(*q)  # Construct a Quaternion object from the tuple
                )

                # Publish the coordinate system of the quadcopter
                publish_quadcopter_coordinate_system(
                    marker_pub=marker_publisher_coord,
                    frame_id="world",
                    position=(info[0]['pos_world'][0], info[0]['pos_world'][1], info[0]['pos_world'][2]),
                    orientation=Quaternion(*q)
                )
                absolute_velocity = np.linalg.norm(info[0]['vel_world'])
                print(f"steps:{info[0]['steps']:<7.2f} ,reward: {info[0]['reward']:<7.2f}, vel_world: {info[0]['vel_world'][0]:<7.2f} {info[0]['vel_world'][1]:<7.2f} {info[0]['vel_world'][2]:<7.2f}, abs_vel: {absolute_velocity} ,PropellerSpeed: {info[0]['propeller_speed'][0]:<7.2f} {info[0]['propeller_speed'][1]:<7.2f} {info[0]['propeller_speed'][2]:<7.2f} {info[0]['propeller_speed'][3]:<7.2f}")

                # Publisher for the MeshResource marker
                mesh_marker.pose.position.x = info[0]['pos_world'][0]
                mesh_marker.pose.position.y = info[0]['pos_world'][1]
                mesh_marker.pose.position.z = info[0]['pos_world'][2]
                mesh_marker.pose.orientation.x = q[0]
                mesh_marker.pose.orientation.y = q[1]
                mesh_marker.pose.orientation.z = q[2]
                mesh_marker.pose.orientation.w = q[3]
                mesh_publisher.publish(mesh_marker)  # Publish the mesh marker

                # Update propeller speeds history
                for i in range(4):
                    prop_speeds_history[i].append(info[0]['propeller_speed'][i])

                # Plot propeller speeds
                plt.clf()  # Clear the current figure
                plt.plot(prop_speeds_history[0], 'r', label='Propeller 1')
                plt.plot(prop_speeds_history[1], 'g', label='Propeller 2')
                plt.plot(prop_speeds_history[2], 'b', label='Propeller 3')
                plt.plot(prop_speeds_history[3], 'y', label='Propeller 4')
                # Set y-axis limits
                plt.ylim(0, 4500)
                # Add labels and legend
                plt.xlabel('Time step')
                plt.ylabel('Propeller speed')
                plt.legend(loc='upper left')
                # Pause the plot
                plt.pause(0.001)

            # Keep the plot open at the end of the loop
            plt.show()


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
