import numpy as np
import sys
import math as m
import time
from scipy.io import loadmat, savemat

sys.path.append('/home/divij/Documents/quadopter/devel/lib')
import custom_robot_wrapper as crw

def main():
    # Load the joint torque data from the input.mat file
    mat_data = loadmat("/home/divij/Documents/quadopter/robot/matlab_torque.mat")
    joint_torque_values = mat_data["matlab_torque"]

    # Define input parameters
    curr_joint_pos = np.array([0.0, 0.0, 0.0], dtype=np.double)
    curr_joint_vel = np.array([0.0, 0.0, 0.0], dtype=np.double)
    curr_joint_acc = np.array([0.0, 0.0, 0.0], dtype=np.double)
    rho = 1000.0
    radius = 0.02
    arm_length = np.array([3.0 , 2.0, 1.0], dtype=np.double)

    # Initialize output arrays
    pos_tcp = np.empty(3, dtype=np.double)

    # Initial time and time step
    t = 0
    dt = 0.001

    joint_pos_py, joint_vel_py, joint_acc_py, joint_torque_py = [], [], [], []
    TCP_pos_py, TCP_vel_py, TCP_acc_py = [], [], []
    start_time = time.time()

    for joint_torque in joint_torque_values:

        joint_torque = np.array(joint_torque, dtype=np.double)
        radius_in = np.array(radius, dtype=np.double)
        arm_length_in = np.array(arm_length, dtype=np.double)
        rho_in = np.array(rho, dtype=np.double)
        # Call the wrapped function
        crw.calc_sys_matrices(curr_joint_pos, curr_joint_vel, rho_in, radius_in, arm_length_in, joint_torque, curr_joint_acc, pos_tcp)

        # Integrating the acceleration to get velocity
        curr_joint_vel = curr_joint_vel + curr_joint_acc * dt
        curr_joint_vel = np.array(curr_joint_vel, dtype=np.double)
        #integrating the velocity to get position
        curr_joint_pos = curr_joint_pos + curr_joint_vel * dt
        curr_joint_pos = np.array(curr_joint_pos, dtype=np.double)

        # Save the joint position, velocity, acceleration, TCP position, velocity, acceleration
        joint_pos_py.append(np.array(curr_joint_pos))
        joint_vel_py.append(np.array(curr_joint_vel))
        joint_acc_py.append(np.array(curr_joint_acc))
        joint_torque_py.append(np.array(joint_torque))

        # Increase the time by the time step
        t += dt

        if t > 5.001:
            break

    end_time = time.time()
    print(f"iteration per sec: {len(joint_pos_py) / (end_time - start_time)}")

    # Save data in a MATLAB file
    output_data = {
        "joint_pos_py": np.array(joint_pos_py),
        "joint_vel_py": np.array(joint_vel_py),
        "joint_acc_py": np.array(joint_acc_py),
        "TCP_pos_py": np.array(TCP_pos_py),
        "TCP_vel_py": np.array(TCP_vel_py),
        "TCP_acc_py": np.array(TCP_acc_py),
        "joint_torque_py": np.array(joint_torque_py),
    }

    print("saving to output.mat...")
    file_path = f"/home/divij/Documents/quadopter/robot/results.mat"
    savemat(file_path, output_data)
    print("saved to output.mat")

if __name__ == "__main__":
    main()
