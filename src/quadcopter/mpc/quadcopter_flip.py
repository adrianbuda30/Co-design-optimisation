import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
N = 100  # Number of time steps
core_mass = 0.05
motor_mass = 0.05
arm_lengths = []
times = []
costs = []


for arm_length in np.arange(0.05, 5.0, 0.05):
    # Define quadcopter dynamics and optimisation
    opti = ca.Opti()

    #arm_length = 0.1  # Distance from the centre to the propellers
    mass = 4 * motor_mass + core_mass
    Ixx = 2 * motor_mass * arm_length ** 2
    Iyy = 2 * motor_mass * arm_length ** 2
    Izz = 4 * motor_mass * arm_length ** 2

    # Time variable
    time_final = opti.variable()
    energy_final = opti.variable()
    opti.set_initial(time_final, 10.0)  # Initial guess
    opti.set_initial(energy_final, 100.0)  # Initial guess
    opti.subject_to(time_final > 0)  # Time must be positive

    # Time discretisation
    time = ca.linspace(0, 1, N) * time_final  # Scaled time vector

    # Control variables (z-force and x, y, z-torques)
    Fz = opti.variable(N)  # Total force in the z direction
    Mx = opti.variable(N)  # Torque about the x-axis
    My = opti.variable(N)  # Torque about the y-axis
    Mz = opti.variable(N)  # Torque about the z-axis
    F_FL = opti.variable(N)  # Total force in the z direction
    F_FR = opti.variable(N)  # Torque about the x-axis
    F_BL = opti.variable(N)  # Torque about the y-axis
    F_BR = opti.variable(N)  # Torque about the z-axis

    # Set initial guesses for the control variables
    opti.set_initial(Fz, mass * 9.81)  # Approx. gravity compensation
    opti.set_initial(Mx, 0)
    opti.set_initial(My, 0)
    opti.set_initial(Mz, 0)

    # State variables
    x_e = opti.variable(N)  # Position in x (inertial frame)
    y_e = opti.variable(N)  # Position in y (inertial frame)
    z_e = opti.variable(N)  # Position in z (inertial frame)
    u_b = opti.variable(N)  # Velocity in x (body frame)
    v_b = opti.variable(N)  # Velocity in y (body frame)
    w_b = opti.variable(N)  # Velocity in z (body frame)
    theta = opti.variable(N)  # Pitch angle
    phi = opti.variable(N)  # Roll angle
    psi = opti.variable(N)  # Yaw angle
    p = opti.variable(N)  # Pitch rate
    q = opti.variable(N)  # Roll rate
    r = opti.variable(N)  # Yaw rate

    # Set initial guesses
    opti.set_initial(x_e, 0)
    opti.set_initial(y_e, 0)
    opti.set_initial(z_e, 0)
    opti.set_initial(u_b, 0)
    opti.set_initial(v_b, 0)
    opti.set_initial(w_b, 0)
    opti.set_initial(theta, 0)
    opti.set_initial(phi, 0)
    opti.set_initial(psi, 0)
    opti.set_initial(p, 0)
    opti.set_initial(q, 0)
    opti.set_initial(r, 0)
    energy_final = 0

    # Dynamics equations
    dt = time_final / (N - 1)  # Time step
    ct = 6
    for k in range(N - 1):
        # Compute individual thruster forces
        F_FL[k] = Fz[k] / 4 - Mx[k] / (2 * arm_length) - My[k] / (2 * arm_length) + Mz[k] / (4 * ct)
        F_FR[k] = Fz[k] / 4 + Mx[k] / (2 * arm_length) - My[k] / (2 * arm_length) - Mz[k] / (4 * ct)
        F_BL[k] = Fz[k] / 4 - Mx[k] / (2 * arm_length) + My[k] / (2 * arm_length) - Mz[k] / (4 * ct)
        F_BR[k] = Fz[k] / 4 + Mx[k] / (2 * arm_length) + My[k] / (2 * arm_length) + Mz[k] / (4 * ct)

        # Physical constraints for thrusters
        opti.subject_to(F_FL[k] >= 0)
        opti.subject_to(F_FR[k] >= 0)
        opti.subject_to(F_BL[k] >= 0)
        opti.subject_to(F_BR[k] >= 0)


        cr, cp, cy = ca.cos(phi[k]), ca.cos(theta[k]), ca.cos(psi[k])
        sr, sp, sy = ca.sin(phi[k]), ca.sin(theta[k]), ca.sin(psi[k])

        # Rotation matrix
        R = ca.vertcat(
            ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            ca.horzcat(-sp, cp * sr, cp * cr)
        )

        # Linear acceleration
        thrust_force = R @ ca.vertcat(0, 0, Fz[k])
        gravity_force = ca.vertcat(0, 0, -mass * 9.81)
        total_force = thrust_force + gravity_force
        linear_acc = total_force / mass

        # Angular acceleration
        angular_acc = ca.vertcat(Mx[k] / Ixx, My[k] / Iyy, Mz[k] / Izz)

        # State update equations (Euler integration)
        opti.subject_to(x_e[k + 1] == x_e[k] + dt * u_b[k])
        opti.subject_to(y_e[k + 1] == y_e[k] + dt * v_b[k])
        opti.subject_to(z_e[k + 1] == z_e[k] + dt * w_b[k])

        opti.subject_to(u_b[k + 1] == u_b[k] + dt * linear_acc[0])
        opti.subject_to(v_b[k + 1] == v_b[k] + dt * linear_acc[1])
        opti.subject_to(w_b[k + 1] == w_b[k] + dt * linear_acc[2])

        opti.subject_to(phi[k + 1] == phi[k] + dt * p[k])
        opti.subject_to(theta[k + 1] == theta[k] + dt * q[k])
        opti.subject_to(psi[k + 1] == psi[k] + dt * r[k])

        opti.subject_to(phi[k] > -np.pi / 4)
        opti.subject_to(theta[k] > -np.pi / 4)
        opti.subject_to(psi[k] > -np.pi / 4)

        opti.subject_to(phi[k] < np.pi / 4)
        opti.subject_to(theta[k] < np.pi / 4)
        opti.subject_to(psi[k] < np.pi / 4)

        opti.subject_to(p[k + 1] == p[k] + dt * angular_acc[0])
        opti.subject_to(q[k + 1] == q[k] + dt * angular_acc[1])
        opti.subject_to(r[k + 1] == r[k] + dt * angular_acc[2])

        energy_final += Fz[k]**2 + Mx[k]**2 + My[k]**2 + Mz[k]**2

    # Initial state constraints
    opti.subject_to([
        x_e[0] == 0,
        y_e[0] == 0,
        z_e[0] == 0,
        u_b[0] == 0,
        v_b[0] == 0,
        w_b[0] == 0,
        theta[0] == 0,
        phi[0] == 0,
        psi[0] == 0,
        p[0] == 0,
        q[0] == 0,
        r[0] == 0,
    ])

    opti.subject_to([
        x_e[N / 4] == 4,
        y_e[N / 4] == 4,
        z_e[N / 4] == 0,
    ])

    opti.subject_to([
        x_e[N / 2] == 8,
        y_e[N / 2] == 0,
        z_e[N / 2] == 0,
    ])

    opti.subject_to([
        x_e[3 * N / 4] == 4,
        y_e[3 * N / 4] == -4,
        z_e[3 * N / 4] == 0,
    ])

    # Final state constraints (target position (5, 5, 5))
    opti.subject_to([
        x_e[-1] == 0,
        y_e[-1] == 0,
        z_e[-1] == 0,
        u_b[-1] == 0,
        v_b[-1] == 0,
        w_b[-1] == 0,
        theta[-1] == 0,
        phi[-1] == 0,
        psi[-1] == 0,
        p[-1] == 0,
        q[-1] == 0,
        r[-1] == 0,
    ])

    # Objective function: minimise time
    opti.minimize(1.0 * time_final + 0.0 * energy_final)

    # Solver setup and solution
    opti.solver("ipopt", {"print_time": True}, {"print_level": 0})
    solution = opti.solve()

    # Extract results
    x_e_opt = solution.value(x_e)
    y_e_opt = solution.value(y_e)
    z_e_opt = solution.value(z_e)
    theta_opt = solution.value(theta)
    phi_opt = solution.value(phi)
    psi_opt = solution.value(psi)
    time_final_opt = solution.value(time_final)
    energy_final_opt = solution.value(energy_final)
    thruster_FL_opt = solution.value(F_FL)
    thruster_FR_opt = solution.value(F_FR)
    thruster_BL_opt = solution.value(F_BL)
    thruster_BR_opt = solution.value(F_BR)

    arm_lengths.append(arm_length)
    times.append(time_final_opt)
    costs.append(energy_final_opt)

    # Results output
    print(f"Move to (5, 5, 5) completed in {time_final_opt:.3f} seconds")



def animate_quadcopter_3d(x_e_vals, y_e_vals, z_e_vals, theta_vals, phi_vals, psi_vals, arm_length=0.5, motor_radius=0.25, interval=50):
    """
    Animates the quadcopter's motion in 3D, including arms, motors, and trajectory.

    Parameters:
        x_e_vals: Array of x positions (world frame)
        y_e_vals: Array of y positions (world frame)
        z_e_vals: Array of z positions (world frame)
        theta_vals: Array of pitch angles (radians)
        phi_vals: Array of roll angles (radians)
        psi_vals: Array of yaw angles (radians)
        arm_length: Length of each arm of the quadcopter
        motor_radius: Radius of the circular motors
        interval: Time interval between frames (milliseconds)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Quadcopter Motion Animation in 3D")

    # Plot trajectory (static)
    ax.plot(x_e_vals, y_e_vals, z_e_vals, label="Trajectory", lw=2, color="blue")
    ax.scatter([x_e_vals[0]], [y_e_vals[0]], [z_e_vals[0]], color="green", label="Start Point", s=100)
    ax.scatter([x_e_vals[-1]], [y_e_vals[-1]], [z_e_vals[-1]], color="red", label="End Point", s=100)

    # Initial quadcopter components
    arm_lines = [ax.plot([], [], [], "b-", lw=2)[0] for _ in range(4)]  # 4 arms
    motor_circles = [ax.plot([], [], [], "r-", lw=1)[0] for _ in range(4)]  # 4 motors

    # Set axis limits
    max_range = np.array([x_e_vals.max() - x_e_vals.min(),
                          y_e_vals.max() - y_e_vals.min(),
                          z_e_vals.max() - z_e_vals.min()]).max() / 2.0

    mid_x = (x_e_vals.max() + x_e_vals.min()) * 0.5
    mid_y = (y_e_vals.max() + y_e_vals.min()) * 0.5
    mid_z = (z_e_vals.max() + z_e_vals.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def update(frame):
        # Current position and angles
        x, y, z = x_e_vals[frame], y_e_vals[frame], z_e_vals[frame]
        theta, phi, psi = theta_vals[frame], phi_vals[frame], psi_vals[frame]

        # Rotation matrix
        cr, cp, cy = np.cos(phi), np.cos(theta), np.cos(psi)
        sr, sp, sy = np.sin(phi), np.sin(theta), np.sin(psi)
        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])

        # Arm endpoints in body frame
        arm_ends = np.array([
            [arm_length / 2, 0, 0],  # Front
            [-arm_length / 2, 0, 0],  # Back
            [0, arm_length / 2, 0],  # Left
            [0, -arm_length / 2, 0]  # Right
        ]).T
        rotated_ends = R @ arm_ends

        # Update arms
        for i, arm_line in enumerate(arm_lines):
            arm_line.set_data([x, x + rotated_ends[0, i]], [y, y + rotated_ends[1, i]])
            arm_line.set_3d_properties([z, z + rotated_ends[2, i]])

        # Update motors
        for i, motor_circle in enumerate(motor_circles):
            motor_centre = np.array([x, y, z]) + rotated_ends[:, i]
            u = np.linspace(0, 2 * np.pi, 100)
            circle_x = motor_radius * np.cos(u)
            circle_y = motor_radius * np.sin(u)
            circle_z = np.zeros_like(u)
            motor_circle_points = np.vstack((circle_x, circle_y, circle_z))
            tilted_motor_circle = R @ motor_circle_points
            tilted_motor_circle[0, :] += motor_centre[0]
            tilted_motor_circle[1, :] += motor_centre[1]
            tilted_motor_circle[2, :] += motor_centre[2]

            motor_circle.set_data(tilted_motor_circle[0, :], tilted_motor_circle[1, :])
            motor_circle.set_3d_properties(tilted_motor_circle[2, :])

        return arm_lines + motor_circles

    # Animation
    anim = FuncAnimation(fig, update, frames=len(x_e_vals), interval=interval, blit=False)
    plt.legend()
    plt.show()

# Example usage
animate_quadcopter_3d(x_e_opt, y_e_opt, z_e_opt, theta_opt, phi_opt, psi_opt, arm_length=0.5, motor_radius=0.25, interval=50)

plt.figure(figsize=(10, 6))

# Plot each thruster force
plt.plot(thruster_FL_opt, label="Front Left (F_FL)", linestyle='-', marker='o')
plt.plot( thruster_FR_opt, label="Front Right (F_FR)", linestyle='-', marker='x')
plt.plot(thruster_BL_opt, label="Back Left (F_BL)", linestyle='-', marker='^')
plt.plot(thruster_BR_opt, label="Back Right (F_BR)", linestyle='-', marker='s')

# Add labels, legend, and grid
plt.xlabel("Time (s)")
plt.ylabel("Thruster Force (N)")
plt.title("Control Inputs: Thruster Forces Over Time")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

plt.figure(figsize=(10, 6))

# Plot each thruster force
plt.plot(theta_opt * 180 / np.pi, label="theta")
plt.plot( phi_opt * 180 / np.pi, label="phi")
plt.plot(psi_opt * 180 / np.pi, label="psi")

# Add labels, legend, and grid
plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.title("Angles Over Time")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

plt.figure(figsize=(10, 6))

# Plot each thruster force
plt.plot(arm_lengths, costs)

# Add labels, legend, and grid
plt.xlabel("Arm length")
plt.ylabel("Cost")
plt.title("Cost distribution")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

plt.figure(figsize=(10, 6))

# Plot each thruster force
plt.plot(arm_lengths, times)

# Add labels, legend, and grid
plt.xlabel("Arm length")
plt.ylabel("Time")
plt.title("Time distribution")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


