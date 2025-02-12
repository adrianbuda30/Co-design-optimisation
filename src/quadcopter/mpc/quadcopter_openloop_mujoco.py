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
A = np.zeros((2*model.nv, 2*model.nv))
B = np.zeros((2*model.nv, model.nu))
epsilon = 1e-6
flg_centered = True

mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

# Define model parameters
N = 100
dt = 0.01

opti = cs.Opti()

# Decision variables
U = opti.variable(model.nu, N)  # Control inputs (nu = number of actuators)
X = opti.variable( 2 * model.nv, N+1)  # State vector (position + velocity)


target_position_numeric = np.array([1.0, 1.0, 1.0])  # Target position

motor_mass = 0.125
core_mass = 0.1
arm_length = 0.0125
mass = 4 * motor_mass + core_mass
Ixx = 2 * motor_mass * arm_length ** 2
Iyy = 2 * motor_mass * arm_length ** 2
Izz = 4 * motor_mass * arm_length ** 2



target_position = opti.parameter(3)
opti.set_value(target_position, target_position_numeric)


# Cost and constraints
target_cost = 0

for i in range(N - 1):
    x_next = X[:, i] + (cs.mtimes(cs.DM(A), X[:, i]) + cs.mtimes(cs.DM(B), U[:, i]))  # Forward simulate
    opti.subject_to(X[:, i+1] == x_next)  # Constrain states to match MuJoCo
    target_cost += cs.sumsqr(X[:3, i + 1] - target_position_numeric) + cs.sumsqr(U[:, i]) * 0.001

# Solver settings
opti.solver("ipopt", {"expand": True, "ipopt.print_level": 0}, {"max_iter": 10000})

final_cost = cs.sumsqr(X[:3, N] - target_position_numeric) * 10
opti.minimize(target_cost + final_cost)

# Initial conditions (set current state as starting point)
x_init = np.hstack((data.qpos[:3], data.qpos[4:], data.qvel))
x_final = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u_init = np.array([9.81 * mass, 0.0, 0.0, 0.0])
opti.subject_to(X[:, 0] == x_init)
opti.subject_to(U[:, 0] == u_init)
opti.subject_to(X[:, -1] == x_final)

# Solve optimization
sol = opti.solve()

# Get optimal control sequence
U_opt = sol.value(U)
X_opt = sol.value(X)

framerate = 60  # Hz
frames = []

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance = 1.0

# Apply optimized controls to MuJoCo
for i in range(N):

    data.ctrl = U_opt[:, i]

    print(data.ctrl, " and", data.qpos[:3])
    mujoco.mj_step(model, data)  # Step the MuJoCo simulator
    renderer.update_scene(data, cam)
    pixels = renderer.render()
    frames.append(pixels)
    #print(data.qpos, "and", data.qvel)

# Main simulation loop for optimization and rendering


fig, ax = plt.subplots()
im = ax.imshow(frames[0])  # Show the first frame

def update(frame_idx):
    im.set_array(frames[frame_idx])  # Update the image for each frame
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / framerate, blit=False)

plt.show()


