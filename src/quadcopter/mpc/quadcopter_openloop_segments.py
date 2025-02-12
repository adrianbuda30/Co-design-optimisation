import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
N = 100  # Number of time steps
core_mass = 0.1

waypoints = [
    (0, 0, 0),
    (-1, -1, 3.5),
    (9, 6, 1),
    (9, -4, 1),
    (-4.5, -6, 3.25),
    (-4.5, -6, 1),
    (4.5, -0.5, 1),
    (-2, 7, 1),
    (-1, -1, 3.5),
]
num_waypoints = len(waypoints)

times = []
costs = []

# Define quadcopter dynamics and optimisation
opti = ca.Opti()


num_segments = num_waypoints - 1
segment_timesteps = N // num_segments  # Divide total horizon equally

# Design variables
arm_length = opti.variable()
motor_mass = opti.variable()


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
opti.set_initial(arm_length, 0.1)  # Initial guess
opti.set_initial(motor_mass, 0.1)  # Initial guess
opti.subject_to(time_final > 0)  # Time must be positive
opti.subject_to(arm_length > 0.05)
opti.subject_to(motor_mass > 0.1)

# Time discretisation
time = ca.linspace(0, 1, N) * time_final  # Scaled time vector

# Create separate trajectory variables for each segment
x_e = [opti.variable(segment_timesteps) for _ in range(num_segments)]
y_e = [opti.variable(segment_timesteps) for _ in range(num_segments)]
z_e = [opti.variable(segment_timesteps) for _ in range(num_segments)]
u_b = [opti.variable(segment_timesteps) for _ in range(num_segments)]
v_b = [opti.variable(segment_timesteps) for _ in range(num_segments)]
w_b = [opti.variable(segment_timesteps) for _ in range(num_segments)]
theta = [opti.variable(segment_timesteps) for _ in range(num_segments)]
phi = [opti.variable(segment_timesteps) for _ in range(num_segments)]
psi = [opti.variable(segment_timesteps) for _ in range(num_segments)]
p = [opti.variable(segment_timesteps) for _ in range(num_segments)]
q = [opti.variable(segment_timesteps) for _ in range(num_segments)]
r = [opti.variable(segment_timesteps) for _ in range(num_segments)]

Fz = [opti.variable(segment_timesteps) for _ in range(num_segments)]
Mx = [opti.variable(segment_timesteps) for _ in range(num_segments)]
My = [opti.variable(segment_timesteps) for _ in range(num_segments)]
Mz = [opti.variable(segment_timesteps) for _ in range(num_segments)]
F_FL = [opti.variable(segment_timesteps) for _ in range(num_segments)]
F_FR = [opti.variable(segment_timesteps) for _ in range(num_segments)]
F_BL = [opti.variable(segment_timesteps) for _ in range(num_segments)]
F_BR = [opti.variable(segment_timesteps) for _ in range(num_segments)]


# Set initial guesses
opti.set_initial(x_e[0][0], 0)
opti.set_initial(y_e[0][0], 0)
opti.set_initial(z_e[0][0], 0)
opti.set_initial(u_b[0][0], 0)
opti.set_initial(v_b[0][0], 0)
opti.set_initial(w_b[0][0], 0)
opti.set_initial(theta[0][0], 0)
opti.set_initial(phi[0][0], 0)
opti.set_initial(psi[0][0], 0)
opti.set_initial(p[0][0], 0)
opti.set_initial(q[0][0], 0)
opti.set_initial(r[0][0], 0)

# Set initial guesses for the control variables
opti.set_initial(Fz[0][0], 0.2 * 9.81)  # Approx. gravity compensation
opti.set_initial(Mx[0][0], 0)
opti.set_initial(My[0][0], 0)
opti.set_initial(Mz[0][0], 0)
opti.set_initial(F_FL[0][0], 0)
opti.set_initial(F_FR[0][0], 0)
opti.set_initial(F_BL[0][0], 0)
opti.set_initial(F_BR[0][0], 0)

energy_final = 0

for i in range(num_segments):

    # Initial conditions: must start at the correct waypoint
    opti.subject_to(x_e[i][0] == waypoints[i][0])
    opti.subject_to(y_e[i][0] == waypoints[i][1])
    opti.subject_to(z_e[i][0] == waypoints[i][2])

    opti.subject_to(x_e[i][-1] == waypoints[i + 1][0])
    opti.subject_to(y_e[i][-1] == waypoints[i + 1][1])
    opti.subject_to(z_e[i][-1] == waypoints[i + 1][2])


# Continuity constraints: ensure final state of one segment = initial state of the next
for i in range(num_segments - 1):
    opti.subject_to(x_e[i][-1] == x_e[i+1][0])
    opti.subject_to(y_e[i][-1] == y_e[i+1][0])
    opti.subject_to(z_e[i][-1] == z_e[i+1][0])
    opti.subject_to(u_b[i][-1] == u_b[i+1][0])
    opti.subject_to(v_b[i][-1] == v_b[i+1][0])
    opti.subject_to(w_b[i][-1] == w_b[i+1][0])
    opti.subject_to(theta[i][-1] == theta[i+1][0])
    opti.subject_to(phi[i][-1] == phi[i+1][0])
    opti.subject_to(psi[i][-1] == psi[i+1][0])
    opti.subject_to(p[i][-1] == p[i+1][0])
    opti.subject_to(q[i][-1] == q[i+1][0])
    opti.subject_to(r[i][-1] == r[i+1][0])
    opti.subject_to(Fz[i][-1] == Fz[i+1][0])
    opti.subject_to(Mx[i][-1] == Mx[i+1][0])
    opti.subject_to(My[i][-1] == My[i+1][0])
    opti.subject_to(Mz[i][-1] == Mz[i+1][0])
    opti.subject_to(F_FL[i][-1] == F_FL[i+1][0])
    opti.subject_to(F_FR[i][-1] == F_FR[i+1][0])
    opti.subject_to(F_BL[i][-1] == F_BL[i+1][0])
    opti.subject_to(F_BR[i][-1] == F_BR[i+1][0])



# Dynamics equations
dt = time_final / (N - 1)  # Time step
ct = 6
for i in range(num_segments):
    for k in range(segment_timesteps - 1):

        # Compute individual thruster forces
        F_FL[i][k] = Fz[i][k] / 4 - Mx[i][k] / (2 * arm_length) - My[i][k] / (2 * arm_length) + Mz[i][k] / (4 * ct)
        F_FR[i][k] = Fz[i][k] / 4 + Mx[i][k] / (2 * arm_length) - My[i][k] / (2 * arm_length) - Mz[i][k] / (4 * ct)
        F_BL[i][k] = Fz[i][k] / 4 - Mx[i][k] / (2 * arm_length) + My[i][k] / (2 * arm_length) - Mz[i][k] / (4 * ct)
        F_BR[i][k] = Fz[i][k] / 4 + Mx[i][k] / (2 * arm_length) + My[i][k] / (2 * arm_length) + Mz[i][k] / (4 * ct)

        # Physical constraints for thrusters
        opti.subject_to(F_FL[i][k] >= 0)
        opti.subject_to(F_FR[i][k] >= 0)
        opti.subject_to(F_BL[i][k] >= 0)
        opti.subject_to(F_BR[i][k] >= 0)


        cr, cp, cy = ca.cos(phi[i][k]), ca.cos(theta[i][k]), ca.cos(psi[i][k])
        sr, sp, sy = ca.sin(phi[i][k]), ca.sin(theta[i][k]), ca.sin(psi[i][k])

        # Rotation matrix
        R = ca.vertcat(
            ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            ca.horzcat(-sp, cp * sr, cp * cr)
        )

        # Linear acceleration
        thrust_force = R @ ca.vertcat(0, 0, Fz[i][k])
        gravity_force = ca.vertcat(0, 0, -mass * 9.81)
        total_force = thrust_force + gravity_force
        linear_acc = total_force / mass

        # Angular acceleration
        angular_acc = ca.vertcat(Mx[i][k] / Ixx, My[i][k] / Iyy, Mz[i][k] / Izz)

        # State update equations (Euler integration)
        opti.subject_to(x_e[i][k + 1] == x_e[i][k] + dt * u_b[i][k])
        opti.subject_to(y_e[i][k + 1] == y_e[i][k] + dt * v_b[i][k])
        opti.subject_to(z_e[i][k + 1] == z_e[i][k] + dt * w_b[i][k])

        opti.subject_to(u_b[i][k + 1] == u_b[i][k] + dt * linear_acc[0])
        opti.subject_to(v_b[i][k + 1] == v_b[i][k] + dt * linear_acc[1])
        opti.subject_to(w_b[i][k + 1] == w_b[i][k] + dt * linear_acc[2])

        opti.subject_to(phi[i][k + 1] == phi[i][k] + dt * p[i][k])
        opti.subject_to(theta[i][k + 1] == theta[i][k] + dt * q[i][k])
        opti.subject_to(psi[i][k + 1] == psi[i][k] + dt * r[i][k])

        opti.subject_to(phi[i][k] > -np.pi / 4)
        opti.subject_to(theta[i][k] > -np.pi / 4)
        opti.subject_to(psi[i][k] > -np.pi / 4)

        opti.subject_to(phi[i][k] < np.pi / 4)
        opti.subject_to(theta[i][k] < np.pi / 4)
        opti.subject_to(psi[i][k] < np.pi / 4)

        opti.subject_to(p[i][k] > -10)
        opti.subject_to(q[i][k] > -10)
        opti.subject_to(r[i][k] > -10)

        opti.subject_to(p[i][k] < 10)
        opti.subject_to(q[i][k] < 10)
        opti.subject_to(r[i][k] < 10)

        opti.subject_to(p[i][k + 1] == p[i][k] + dt * angular_acc[0])
        opti.subject_to(q[i][k + 1] == q[i][k] + dt * angular_acc[1])
        opti.subject_to(r[i][k + 1] == r[i][k] + dt * angular_acc[2])

        energy_final += Fz[i][k]**2 + Mx[i][k]**2 + My[i][k]**2 + Mz[i][k]**2


# Objective function: minimise time
opti.minimize(0.99 * time_final + 0.01 * energy_final)

# Solver setup and solution
opti.solver("ipopt", {"print_time": True}, {"print_level": 0})

solution = opti.solve()


# Extract results
x_e_opt = np.concatenate([solution.value(x_e[i]) for i in range(num_segments)])
y_e_opt = np.concatenate([solution.value(y_e[i]) for i in range(num_segments)])
z_e_opt = np.concatenate([solution.value(z_e[i]) for i in range(num_segments)])
theta_opt = np.concatenate([solution.value(theta[i]) for i in range(num_segments)])
phi_opt = np.concatenate([solution.value(phi[i]) for i in range(num_segments)])
psi_opt = np.concatenate([solution.value(psi[i]) for i in range(num_segments)])

thruster_FL_opt = np.concatenate([solution.value(F_FL[i]) for i in range(num_segments)])
thruster_FR_opt = np.concatenate([solution.value(F_FR[i]) for i in range(num_segments)])
thruster_BL_opt = np.concatenate([solution.value(F_BL[i]) for i in range(num_segments)])
thruster_BR_opt = np.concatenate([solution.value(F_BR[i]) for i in range(num_segments)])

time_final_opt = solution.value(time_final)
energy_final_opt = solution.value(energy_final)

arm_length_opt = solution.value(arm_length)
motor_mass_opt = solution.value(motor_mass)

# Results output
print(f"Move to final position completed in {time_final_opt:.3f} seconds, with an arm length of: {arm_length_opt} and a motor mass of: {motor_mass_opt}")


def animate_quadcopter_3d(x_e_vals, y_e_vals, z_e_vals, theta_vals, phi_vals, psi_vals, arm_length=0.1, motor_radius=0.1, interval=50):
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

    for i in range(num_waypoints):
        ax.scatter([waypoints[i][0]], [waypoints[i][1]], [waypoints[i][2]], color="green", label="Gate", s=100)

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

plt.figure(figsize=(10, 8))
# Plot trajectory (static)
plt.plot(x_e_opt, y_e_opt, label="Trajectory", lw=2, color="blue")
for i in range(num_waypoints):
    plt.scatter([waypoints[i][0]], [waypoints[i][1]], color="green", label="Gate", s=100)


plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.show()

plt.figure(figsize=(10, 8))
# Plot trajectory (static)
plt.plot(x_e_opt, z_e_opt, label="Trajectory", lw=2, color="blue")
for i in range(num_waypoints):
    plt.scatter([waypoints[i][0]], [waypoints[i][2]], color="green", label="Gate", s=100)


plt.xlabel("X (m)")
plt.ylabel("Z (m)")

plt.show()



