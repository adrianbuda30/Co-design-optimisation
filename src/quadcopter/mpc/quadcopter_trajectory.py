import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
g = 9.81  # Gravity (m/s^2)
m = 1.0   # Mass of the quadcopter (kg)

# Time horizon
T = ca.MX.sym("T")  # Total time (to be optimised)
N = 50  # Number of discretisation points

# Decision variables
dt = T / N  # Time step
x = ca.MX.sym("x", 6, N + 1)  # States: [x, y, z, vx, vy, vz]
u = ca.MX.sym("u", 3, N)  # Controls: [ux, uy, uz]

# Dynamics (discretised)
x_next = []
for i in range(N):
    x_next_i = x[:, i] + dt * ca.vertcat(
        x[3, i],  # dx/dt = vx
        x[4, i],  # dy/dt = vy
        x[5, i],  # dz/dt = vz
        u[0, i] / m,  # dvx/dt = ux / m
        u[1, i] / m,  # dvy/dt = uy / m
        u[2, i] / m - g  # dvz/dt = uz / m - g
    )
    x_next.append(x_next_i)
x_next = ca.horzcat(*x_next)

# Cost function (time and energy)
cost_time = T
cost_energy = ca.sum2(u ** 2 * dt)  # Quadratic cost on controls (energy)
cost = cost_time + 0.01 * cost_energy  # Weight energy cost

# Constraints
constraints = []
constraints += [x[:, 1:] - x_next]  # Dynamics constraints
constraints += [x[:, 0] - ca.vertcat(0, 0, 0, 0, 0, 0)]  # Initial state
constraints += [x[:, -1] - ca.vertcat(5, 5, 5, 0, 0, 0)]  # Final state
constraints += [u]  # No bounds yet on controls

# Bounds
x_lb = -ca.inf * ca.DM.ones(6, N + 1)
x_ub = ca.inf * ca.DM.ones(6, N + 1)
u_lb = -10.0 * ca.DM.ones(3, N)
u_ub = 10.0 * ca.DM.ones(3, N)
T_lb = 0.1
T_ub = 10.0

# NLP problem
w = ca.vertcat(x.reshape((-1, 1)), u.reshape((-1, 1)), T)
lbw = ca.vertcat(x_lb.reshape((-1, 1)), u_lb.reshape((-1, 1)), T_lb)
ubw = ca.vertcat(x_ub.reshape((-1, 1)), u_ub.reshape((-1, 1)), T_ub)
g = ca.vertcat(*constraints)
lbg = ca.DM.zeros(g.shape[0])
ubg = ca.DM.zeros(g.shape[0])

nlp = {"x": w, "f": cost, "g": g}
solver = ca.nlpsol("solver", "ipopt", nlp)

# Solve
sol = solver(
    x0=ca.DM.zeros(w.shape[0]),  # Initial guess
    lbx=lbw,
    ubx=ubw,
    lbg=lbg,
    ubg=ubg,
)

# Extract solution
w_opt = sol["x"]
x_opt = w_opt[:6 * (N + 1)].reshape((6, N + 1))
u_opt = w_opt[6 * (N + 1):6 * (N + 1) + 3 * N].reshape((3, N))
T_opt = w_opt[-1]

# Display results
print("Optimal time: ", T_opt)
print("Optimal trajectory: ", x_opt)
print("Optimal controls: ", u_opt)


# Extract states for plotting
x_vals = x_opt[0, :].full().flatten()
y_vals = x_opt[1, :].full().flatten()
z_vals = x_opt[2, :].full().flatten()
u_vals = u_opt.full()

# Plot the trajectory in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, label="Optimal Trajectory", color="blue", lw=2)
ax.scatter(0, 0, 0, color="green", label="Start Point (A)", s=100)
ax.scatter(5, 5, 5, color="red", label="End Point (B)", s=100)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("Optimal Quadcopter Trajectory")
ax.legend()
plt.show()

# Plot control inputs
time_vals = [T_opt.full().item() * i / N for i in range(N)]

plt.figure(figsize=(10, 6))
plt.plot(time_vals, u_vals[0, :], label="Control Input ux", lw=2)
plt.plot(time_vals, u_vals[1, :], label="Control Input uy", lw=2)
plt.plot(time_vals, u_vals[2, :], label="Control Input uz", lw=2)
plt.axhline(10, color="red", linestyle="--", label="Upper Bound (10 N)")
plt.axhline(-10, color="red", linestyle="--", label="Lower Bound (-10 N)")
plt.xlabel("Time (s)")
plt.ylabel("Control Force (N)")
plt.title("Control Inputs Over Time")
plt.legend()
plt.grid()
plt.show()
