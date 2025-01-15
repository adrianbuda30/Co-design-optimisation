import numpy as np
import matplotlib.pyplot as plt

import casadi as ca
import numpy as np

class QuadcopterNMPC:
    def __init__(self, dt, horizon, max_thrust, max_torque, mass, g, I):
        self.dt = dt
        self.horizon = horizon
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.mass = mass
        self.g = g
        self.I = I

        # State: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.n_states = 12
        # Control: [thrust, torque_x, torque_y, torque_z]
        self.n_controls = 4

    def _rk4(self, state, control):
        """Runge-Kutta integration for dynamics."""
        def dynamics(state, control):
            pos = state[:3]
            vel = state[3:6]
            angles = state[6:9]
            angular_rates = state[9:12]
            thrust, torques = control[0], control[1:]

            # Rotation matrix
            cr, cp, cy = ca.cos(angles[0]), ca.cos(angles[1]), ca.cos(angles[2])
            sr, sp, sy = ca.sin(angles[0]), ca.sin(angles[1]), ca.sin(angles[2])
            R = ca.vertcat(
                ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                ca.horzcat(-sp, cp * sr, cp * cr)
            )

            # Forces and accelerations
            thrust_force = R @ ca.vertcat(0, 0, thrust * self.max_thrust)
            gravity_force = ca.vertcat(0, 0, -self.mass * self.g)
            total_force = thrust_force + gravity_force
            linear_acc = total_force / self.mass

            # Torques and angular accelerations
            torques = torques * self.max_torque
            angular_acc = torques.T / self.I

            print(vel)
            print(linear_acc)
            print(angular_rates)
            print(angular_acc)

            # Combine accelerations
            acc = ca.vertcat(
                vel,  # dx/dt = vel
                linear_acc,  # dv/dt = acc
                angular_rates,  # dangles/dt = angular_rates
                angular_acc  # dangular_rates/dt = angular_acc
            )
            return acc

        k1 = dynamics(state, control)
        k2 = dynamics(state + 0.5 * self.dt * k1, control)
        k3 = dynamics(state + 0.5 * self.dt * k2, control)
        k4 = dynamics(state + self.dt * k3, control)
        return state + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def setup_mpc(self, x0, x_ref):
        # Optimization variables
        X = ca.SX.sym('X', self.horizon + 1, self.n_states)  # States
        U = ca.SX.sym('U', self.horizon, self.n_controls)  # Controls

        # Objective and constraints
        cost = 0
        g = []

        # Initial state constraint
        g.append(X[0, :] - x0)

        # Cost function and dynamic constraints
        for t in range(self.horizon):
            # Tracking cost
            cost += ca.mtimes([(X[t, :] - x_ref), (X[t, :] - x_ref).T])

            # Control effort cost
            cost += ca.mtimes([U[t, :].T, U[t, :]])

            # Dynamics constraint
            x_next = self._rk4(X[t, :], U[t, :])
            g.append(X[t + 1, :] - x_next)

        # Terminal cost
        cost += ca.mtimes([(X[-1, :] - x_ref).T, X[-1, :] - x_ref])

        # Decision variables
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Constraints vector
        g = ca.vertcat(*g)

        # Define problem
        nlp = {'x': opt_vars, 'f': cost, 'g': g}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        return solver

    def solve(self, solver, x0, x_ref):
        # Set initial guess and bounds
        n_vars = self.horizon * (self.n_states + self.n_controls) + self.n_states
        lbx = -ca.inf * np.ones(n_vars)
        ubx = ca.inf * np.ones(n_vars)

        lbg = np.zeros(self.horizon * self.n_states)
        ubg = np.zeros(self.horizon * self.n_states)

        # Solve the optimization problem
        sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        opt_vals = sol['x']

        # Extract solution
        X_sol = np.array(opt_vals[:self.horizon * self.n_states]).reshape((self.horizon, self.n_states))
        U_sol = np.array(opt_vals[self.horizon * self.n_states:]).reshape((self.horizon, self.n_controls))

        return X_sol, U_sol

# Define the circular trajectory
def circular_trajectory(t, radius=1.0, omega=1.0, z_const=1.0):
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)
    z = z_const
    return np.array([x, y, z])

# Simulate and track the circular trajectory
def simulate_circular_trajectory(nmpc, solver, x0, T, radius=1.0, omega=1.0, z_const=1.0):
    dt = nmpc.dt
    steps = int(T / dt)
    trajectory = []
    controls = []

    state = x0
    for step in range(steps):
        # Compute the reference state at the current time
        t = step * dt
        pos_ref = circular_trajectory(t, radius, omega, z_const)
        x_ref = np.zeros((1, nmpc.n_states))
        x_ref[0,:3] = pos_ref  # Set position reference

        # Solve NMPC
        X_sol, U_sol = nmpc.solve(solver, state, x_ref)
        control = U_sol[0, :]  # Use the first control action

        # Apply the control (simulate one step using RK4)
        state = nmpc._rk4(state, control).full().flatten()

        # Store trajectory and controls
        trajectory.append(state)
        controls.append(control)

    return np.array(trajectory), np.array(controls)

# Simulation parameters
dt = 0.05
horizon = 10
T = 10.0  # Total simulation time
radius = 1.0
omega = 2 * np.pi / T  # One full revolution in T seconds
z_const = 1.0

# Quadcopter parameters
max_thrust = 10.0
max_torque = 1.0
mass = 1.0
g = 9.81
I = np.array([0.1, 0.1, 0.2])

# Initialize NMPC
nmpc = QuadcopterNMPC(dt, horizon, max_thrust, max_torque, mass, g, I)

# Initial state
x0 = np.zeros((1, nmpc.n_states))

# Set up the solver
solver = nmpc.setup_mpc(x0, x_ref=np.zeros((1, nmpc.n_states)))  # Placeholder x_ref

# Run simulation
trajectory, controls = simulate_circular_trajectory(nmpc, solver, x0, T, radius, omega, z_const)

# Plot results
trajectory = np.array(trajectory)
time = np.arange(0, T, dt)

plt.figure(figsize=(10, 6))
ax = plt.axes(projection="3d")
ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Actual Trajectory")
circle = np.array([circular_trajectory(t, radius, omega, z_const) for t in time])
ax.plot3D(circle[:, 0], circle[:, 1], circle[:, 2], label="Reference Trajectory", linestyle="--")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.title("Quadcopter Circular Trajectory Tracking")
plt.show()