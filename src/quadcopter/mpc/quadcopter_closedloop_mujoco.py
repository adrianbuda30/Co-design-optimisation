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

class MujocoWrapper:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(self.model)

        # Create a camera instance
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)

        # Adjust camera settings
        self.cam.lookat[:] = self.data.qpos[:3]  # Focus on quadcopter position
        self.cam.distance = 5.0  # Reduce the distance to move closer
        self.cam.elevation = -20  # Adjust angle if needed
        self.cam.azimuth = 90  # Adjust horizontal rotation


    def get_qpos(self):
        return self.data.qpos[:]

    def render(self):
        mujoco.mj_forward(self.model, self.data)

        # Update camera position to follow the quadcopter
        #self.cam.lookat[:] = self.data.qpos[:3]
        self.cam.distance = 5.0


        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_ctrl(self, ctrl):
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)



wrapper = MujocoWrapper(model)

# Define model parameters
n_pos = 7
n_vel = 6
n_control = 4
N = 100
dt = 0.004

target_position_numeric = np.array([1.0, 1.0, 1.0])  # Target position

motor_mass = 0.125
core_mass = 0.1
arm_length = 0.0125
mass = 4 * motor_mass + core_mass
Ixx = 2 * motor_mass * arm_length ** 2
Iyy = 2 * motor_mass * arm_length ** 2
Izz = 4 * motor_mass * arm_length ** 2

# CasADi Opti stack
opti = cs.Opti()

# Decision variables
q = opti.variable(n_pos, N+1)
q_dot = opti.variable(n_vel, N)
u = opti.variable(n_control, N)



# Parameters
q0 = opti.parameter(n_pos)
q_dot0 = opti.parameter(n_vel)
u0 = opti.parameter(n_control)

# Initial conditions
q0_numeric = np.zeros(n_pos)
q_dot0_numeric = np.zeros(n_vel)
u0_numeric = np.zeros(n_control)

opti.set_value(q0, q0_numeric)
opti.set_value(q_dot0, q_dot0_numeric)
opti.set_value(u0, u0_numeric)


target_position = opti.parameter(3)
opti.set_value(target_position, target_position_numeric)

# Set initial conditions
opti.subject_to(q[:, 0] == q0)
opti.subject_to(q_dot[:, 0] == q_dot0)
opti.subject_to(u[:, 0] == u0)

# Cost and constraints
target_cost = 0
for i in range(N - 1):
    cr, cp, cy = cs.cos(q[4, i]), cs.cos(q[5, i]), cs.cos(q[6, i])
    sr, sp, sy = cs.sin(q[4, i]), cs.sin(q[5, i]), cs.sin(q[6, i])

    # Rotation matrix
    R = cs.vertcat(
        cs.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        cs.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        cs.horzcat(-sp, cp * sr, cp * cr)
    )

    # Linear acceleration
    thrust_force = R @ cs.vertcat(0, 0, u[0, i])
    gravity_force = cs.vertcat(0, 0, -mass * 9.81)
    total_force = thrust_force + gravity_force
    linear_acc = total_force / mass
    angular_acc = cs.vertcat(u[1, i] / Ixx, u[2, i] / Iyy, u[3, i] / Izz)

    # Dynamics (Euler integration)
    q_next = cs.vertcat(q[:3, i] + q_dot[:3, i] * dt, q[3, i], q[4:, i] + q_dot[3:, i] * dt)
    q_next[4:] = np.mod(q_next[4:] + np.pi, 2 * np.pi) - np.pi
    qdot_next = cs.vertcat(q_dot[:3, i] + linear_acc * dt, q_dot[3:, i] + angular_acc * dt)
    opti.subject_to(q[:, i+1] == q_next)
    opti.subject_to(q_dot[:, i+1] == qdot_next)



    #opti.subject_to(opti.bounded(0, u[:, i], 10))
    #opti.subject_to(opti.bounded(-np.pi / 2, q[4, i], np.pi / 2))
    #opti.subject_to(opti.bounded(-np.pi / 2, q[5, i], np.pi / 2))
    #opti.subject_to(opti.bounded(-np.pi / 2, q[6, i], np.pi / 2))


    target_cost += cs.sumsqr(q[:3, i] - target_position_numeric) + cs.sumsqr(u[:, i]) * 0.001


# Minimise total cost
terminal_cost = cs.sumsqr(q[:3, N] - target_position_numeric) * 10  # Increase weight as needed
opti.minimize(target_cost + terminal_cost)

# Solver settings
opti.solver("ipopt", {"expand": True, "ipopt.print_level": 0}, {"max_iter": 100000})

# Simulation loop
duration = 1.0 # seconds
framerate = 60  # Hz
frames = []
mujoco.mj_resetData(wrapper.model, wrapper.data)

# Main simulation loop for optimization and rendering

while wrapper.data.time < duration:


    sol = opti.solve()
    sol_q = sol.value(q)
    sol_q_dot = sol.value(q_dot)
    sol_u = sol.value(u)

    wrapper.step()
    wrapper.set_ctrl(sol_u[:, 0])

    if len(frames) < wrapper.data.time * framerate:
        # Adjust camera settings
        pixels = wrapper.render()
        frames.append(pixels)


    opti.set_value(u0, sol_u[:, 1])

    q0_numeric = wrapper.data.qpos.flat.copy()
    opti.set_value(q0, q0_numeric)

    q_dot0_numeric = wrapper.data.qvel.flat.copy()
    opti.set_value(q_dot0, q_dot0_numeric)



    print(sol_u[:, 0])
    print(sol_q[:, 0])


    if len(frames) == 0:
        raise ValueError("No frames were generated for the animation. Check the rendering process.")



fig, ax = plt.subplots()
im = ax.imshow(frames[0])  # Show the first frame

def update(frame_idx):
    im.set_array(frames[frame_idx])  # Update the image for each frame
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / framerate, blit=False)

plt.show()


