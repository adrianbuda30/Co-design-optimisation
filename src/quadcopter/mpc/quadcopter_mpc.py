import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

class QuadcopterNMPC:
    def __init__(self, dt, max_thrust, max_torque, mass, I, g, horizon=50):
        self.dt = dt
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.mass = mass
        self.I = I
        self.g = g
        self.state = np.zeros(12)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.horizon = horizon

        # Trajectory definition (circular trajectory)
        radius = 8
        height_amplitude = 0
        center = np.array([0, 8, 0])  # Center of circle
        num_points = 500  # Number of trajectory points
        t = np.linspace(0, 4 * np.pi, num_points)  # time or angle variable

        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = height_amplitude * np.cos(t)

        self.trajectory = np.vstack((x, y, z)).T
        self.current_index = 0

        # CasADi setup
        self.opti = ca.Opti()
        self._setup_nmpc()

    def _setup_nmpc(self):
        """Set up the CasADi NMPC problem."""
        # Decision variables
        self.X = self.opti.variable(12, self.horizon + 1)  # States
        self.U = self.opti.variable(4, self.horizon)  # Controls (thrust + torques)

        # Parameters
        self.X0 = self.opti.parameter(12)  # Initial state
        self.ref = self.opti.parameter(3, self.horizon)  # Reference trajectory

        # Cost function
        cost = 0
        for t in range(self.horizon):
            # Position error
            pos_error = self.X[:3, t] - self.ref[:, t]
            cost += ca.dot(pos_error, pos_error)
            # Velocity error
            vel_error = self.X[3:6, t]
            cost += 0.1 * ca.dot(vel_error, vel_error)
            # Input cost
            cost += 0.01 * ca.dot(self.U[:, t], self.U[:, t])
        self.opti.minimize(cost)

        # Dynamics constraints (Runge-Kutta)
        for t in range(self.horizon):
            x_next = self.X[:, t] + self._rk4(self.X[:, t], self.U[:, t]) * self.dt
            self.opti.subject_to(self.X[:, t + 1] == x_next)

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.X0)  # Initial state constraint
        self.opti.subject_to(0 <= self.U[0, :])  # Thrust bounds
        self.opti.subject_to(self.U[0, :] <= 1)  # Thrust bounds
        self.opti.subject_to(0 <= self.U[1, :])  # Torque bounds
        self.opti.subject_to(self.U[1, :] <= 1)  # Torque bounds
        self.opti.subject_to(0 <= self.U[2, :])  # Torque bounds
        self.opti.subject_to(self.U[2, :] <= 1)  # Torque bounds
        self.opti.subject_to(0 <= self.U[3, :])  # Torque bounds
        self.opti.subject_to(self.U[3, :] <= 1)  # Torque bounds

        # Solver setup
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
        self.opti.solver("ipopt", opts)

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
            angular_acc = torques / self.I

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
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def get_reference(self):
        """Get the next reference points on the trajectory."""

        distances = np.zeros(500)
        for i in range(500):
            distances[i] = np.linalg.norm(self.trajectory[i,:] - self.state[:3])
        self.current_index = np.argmin(distances)
        end_index = min(self.current_index + self.horizon, len(self.trajectory))
        ref_points = self.trajectory[self.current_index:end_index]

        # Pad if not enough reference points
        if ref_points.shape[0] < self.horizon:
            padding = np.tile(ref_points[-1], (self.horizon - ref_points.shape[0], 1))
            ref_points = np.vstack([ref_points, padding])
        return ref_points.T

    def step(self):
        """Perform one NMPC step."""
        # Update parameters
        self.opti.set_value(self.X0, self.state)
        ref = self.get_reference()
        self.opti.set_value(self.ref, ref)

        # Solve NMPC
        sol = self.opti.solve()
        u_opt = sol.value(self.U[:, 0])

        # Apply dynamics
        self.state += self._rk4(self.state, u_opt) * self.dt

        # Return state and reference for debugging
        return self.state, ref[:, 0]

# Parameters
dt = 0.01
max_thrust = 10
max_torque = 10
mass = 0.1
I = np.array([0.1, 0.1, 0.15])  # Moments of inertia
g = 9.81

# Initialize and run the simulation
controller = QuadcopterNMPC(dt, max_thrust, max_torque, mass, I, g)

states = []
references = []

for index in range(500):  # Simulate 1000 steps
    state, ref = controller.step()
    states.append(state[:3])  # Only collect position [x, y, z]
    references.append(ref)
    print(index)

states = np.array(states)
references = np.array(references)

# Plot the trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:,2], label="Quadcopter Trajectory", color='b')
ax.plot(references[:, 0], references[:, 1], references[:, 2], label="Reference Trajectory", color='r', linestyle='--')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Spiral Trajectory')
ax.legend()
ax.grid(True)
plt.show()
