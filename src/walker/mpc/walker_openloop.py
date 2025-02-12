import mujoco
from adam.casadi import KinDynComputations
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

matplotlib.use("TkAgg")  # Ensure correct backend

# Load model and setup
model = mujoco.MjModel.from_xml_path(f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/assets/walker2d.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

class MujocoWrapper:
    def __init__(self, model, exclude_current_positions_from_observation=False):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(self.model)
        self.exclude_current_positions_from_observation = exclude_current_positions_from_observation

        # Create a camera instance
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)

        # Adjust camera settings
        self.cam.lookat[:] = self.data.qpos[:3]  # Focus on quadcopter position
        self.cam.distance = 2.0  # Reduce the distance to move closer
        self.cam.elevation = -20  # Adjust angle if needed
        self.cam.azimuth = 90  # Adjust horizontal rotation


    def get_qpos(self):
        return self.data.qpos[:]

    def render(self):
        mujoco.mj_forward(self.model, self.data)

        # Update camera position to follow the quadcopter
        self.cam.lookat[:] = self.data.qpos[:3]
        self.cam.distance = 2.0


        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_ctrl(self, ctrl):
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)


wrapper = MujocoWrapper(model)

# Define model parameters
n_pos = 9
n_vel = 9
n_joints = 6
N = 100
dt = 0.001
target_pos_numeric = 5.0  # Target position in the x-direction

# CasADi Opti stack
opti = cs.Opti()

# Decision variables
q = opti.variable(n_pos, N+1)
q_dot = opti.variable(n_vel, N)
u = opti.variable(n_joints, N)

# Parameters
q0 = opti.parameter(n_pos)
q_dot0 = opti.parameter(n_vel)
u0 = opti.parameter(n_joints)

# Initial conditions
q0_numeric = np.zeros(n_pos)
q_dot0_numeric = np.zeros(n_vel)
u0_numeric = 0.1 * np.ones(n_joints)

opti.set_value(q0, q0_numeric)
opti.set_value(q_dot0, q_dot0_numeric)
opti.set_value(u0, u0_numeric)

target_pos = opti.parameter(1)
opti.set_value(target_pos, target_pos_numeric)

# Solver settings
opti.solver("ipopt", {"expand": True, "ipopt.print_level": 0}, {"max_iter": 100})


# Set initial conditions
opti.subject_to(q[:, 0] == q0)
opti.subject_to(q_dot[:, 0] == q_dot0)
opti.subject_to(u[:, 0] == u0)

# Set initial conditions
opti.subject_to(q[:, -1] == np.array([5, 0, 0, 0, 0, 0, 0, 0, 0]))


frames = []
mujoco.mj_resetData(wrapper.model, wrapper.data)

# Solver setup and solution
opti.solver("ipopt", {"print_time": True}, {"print_level": 0})

sol = opti.solve()
sol_q = sol.value(q)
sol_q_dot = sol.value(q_dot)
sol_u = sol.value(u)

# Cost and constraints
target_cost = 0
for i in range(N - 1):
    wrapper.set_ctrl(sol_u[:, 0])
    wrapper.step()

    q_next = wrapper.data.qpos.flat.copy()
    qdot_next = wrapper.data.qvel.flat.copy()

    opti.subject_to(q[:, i+1] == q_next)
    opti.subject_to(q_dot[:, i+1] == qdot_next)

    wrapper.renderer.update_scene(wrapper.data)
    pixels = wrapper.renderer.render()
    frames.append(pixels)

    sol = opti.solve()
    sol_q = sol.value(q)
    sol_q_dot = sol.value(q_dot)
    sol_u = sol.value(u)

    print(sol_q, "and", sol_u)

    # Centre of mass velocity (COM velocity is estimated from q_dot)
    current_pos = q[0, i+1]  # x-component
    target_cost += (current_pos - target_pos) ** 2 + cs.sumsqr(u[:, i]) * 1e-5  # Penalise deviation from target position

# Minimise total cost
opti.minimize(target_cost)

fig, ax = plt.subplots()
im = ax.imshow(frames[0])  # Show the first frame

def update(frame_idx):
    im.set_array(frames[frame_idx])  # Update the image for each frame
    return [im]


framerate = 10
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / framerate, blit=False)

plt.show()



