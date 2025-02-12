import mujoco
from adam.casadi import KinDynComputations
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

matplotlib.use("TkAgg")  # Ensure correct backend

# Load model and setup
model = mujoco.MjModel.from_xml_path(f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/quadcopter/assets/quadcopter.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# Define model parameters
n_pos = 7
n_vel = 6
n_control = 4
N = 100
dt = 0.01

opti = cs.Opti()

# Decision variables
U = opti.variable(model.nu, N)  # Control inputs (nu = number of actuators)
X = opti.variable(model.nq + model.nv, N+1)  # State vector (position + velocity)


target_position_numeric = np.array([0.0, 0.0, 1.0])  # Target position

motor_mass = 0.125
core_mass = 0.1
arm_length = 0.0125
mass = 4 * motor_mass + core_mass
Ixx = 2 * motor_mass * arm_length ** 2
Iyy = 2 * motor_mass * arm_length ** 2
Izz = 4 * motor_mass * arm_length ** 2



target_position = opti.parameter(3)
opti.set_value(target_position, target_position_numeric)

def mujoco_step(control):
    """ Simulates MuJoCo forward one step given state and control """
    data.ctrl = control.flatten()

    # Step the simulation
    mujoco.mj_step(model, data)

    # Get new state
    new_state = np.hstack((data.qpos.flatten(), data.qvel.flatten()))
    return new_state


framerate = 60  # Hz
frames = []

# Cost and constraints
target_cost = 0

for i in range(N):
    x_next = mujoco_step(cs.full(U[:, i]))  # Forward simulate
    opti.subject_to(X[:, i+1] == x_next)  # Constrain states to match MuJoCo
    target_cost += cs.sumsqr(X[:3, i] - target_position_numeric) + cs.sumsqr(U[:, i]) * 0.001

# Solver settings
opti.solver("ipopt", {"expand": True, "ipopt.print_level": 0}, {"max_iter": 10000})

opti.minimize(target_cost)

# Initial conditions (set current state as starting point)
x_init = np.hstack((data.qpos, data.qvel))
opti.subject_to(X[:, 0] == x_init)

# Solve optimization
sol = opti.solve()

# Get optimal control sequence
U_opt = sol.value(U)

# Apply optimized controls to MuJoCo
for i in range(N):
    data.ctrl = U_opt[:, i]
    mujoco.mj_step(model, data)  # Step the MuJoCo simulator

# Main simulation loop for optimization and rendering


fig, ax = plt.subplots()
im = ax.imshow(frames[0])  # Show the first frame

def update(frame_idx):
    im.set_array(frames[frame_idx])  # Update the image for each frame
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / framerate, blit=False)

plt.show()


