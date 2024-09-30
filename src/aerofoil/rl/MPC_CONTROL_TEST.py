import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy as cp

import math as m

m = 1.5
x_cg = 0.62
x_ea = 0.35
k_h = 1800
k_a = 370
k_b = 390

rho = 1.225
c = 0.254
t = 0.12

m_alpha = 0.8 * m
m_beta = 0.2 * m

x_cg = x_cg * c
x_ea = x_ea * c

x_ac = 0.25 * c
x_fh = 0.88 * c
x_h = 0.80 * c
r_alpha = x_cg - x_ea
r_beta = x_fh - x_h
r = x_h - x_ea
Na = 2
Vinf = 75
W_g = 0.1 * Vinf
a = np.array([0.165, 0.335])
b = np.array([0.0455, 0.3])
a_g = np.array([0.5, 0.5])
b_g = np.array([0.13, 1.0])
Ng = 2

nu_ea = (x_ea - c / 2) / (c / 2)
nu_fh = (x_h - c / 2) / (c / 2)
theta_fh = np.arccos(-nu_fh)

# Aerodynamics influence coefficient matrices
CLa0qs = 2 * np.pi
CLa1qs = 2 * np.pi * (1 / 2 - nu_ea)
CMa1qs = -np.pi / 4
CLa1nc = np.pi
CLa2nc = -np.pi * nu_ea
CMa1nc = -np.pi / 4
CMa2nc = -np.pi / 4 * (1 / 4 - nu_ea)

CLd0qs = 2 * np.pi - 2 * theta_fh + 2 * np.sin(theta_fh)
CLd1qs = (1 / 2 - nu_fh) * (2 * np.pi - 2 * theta_fh) + (2 - nu_fh) * np.sin(theta_fh)
CMd0qs = -(1 / 2) * (1 + nu_fh) * np.sin(theta_fh)
CMd1qs = -(1 / 4) * (np.pi - nu_fh + (2 / 3) * (1 / 2 - nu_fh) * (2 + nu_fh) * np.sin(theta_fh))
CLd1nc = np.pi - theta_fh - nu_fh * np.sin(theta_fh)
CLd2nc = -nu_fh * (np.pi - theta_fh) + (1 / 3) * (2 + nu_fh ** 2) * np.sin(theta_fh)
CMd1nc = -(1 / 4) * (np.pi - theta_fh + (2 / 3 - nu_fh - (2 / 3) * nu_fh ** 2) * np.sin(theta_fh))
CMd2nc = -(1 / 4) * ((1 / 4 - nu_fh) * (np.pi - theta_fh) + (
            2 / 3 - (5 / 12) * nu_fh + (1 / 3) * nu_fh ** 2 + (1 / 6) * nu_fh ** 3) * np.sin(theta_fh))

# Equation (3.60)
A_0 = np.array([[0, CLa0qs, CLd0qs],
                [0, 0, CMd0qs]])
A_1 = np.array([[CLa0qs, CLa1qs + CLa1nc, CLd1qs + CLd1nc],
                [0, CMa1qs + CMa1nc, CMd1qs + CMd1nc]])
A_2 = np.array([[CLa1nc, CLa2nc, CLd2nc],
                [CMa1nc, CMa2nc, CMd2nc]])
A_3 = np.array([[CLa0qs, CLa1qs, CLd1qs],
                [0, 0, 0]])

Ns = 3
A_a = np.zeros((Ns * Na, Ns * Na))
B_a = np.zeros((Ns * Na, Ns))
C_a = np.zeros((Ng, Ns * Na))

A_g = np.zeros((Ng, Ng))
B_g = np.zeros((Ng, 1))
C_g = np.zeros((Ng, 2))

# State-space matrices for unsteady aero - Equation (3.69)
for j in range(1, Na + 1):
    A_a[(j - 1) * 3:j * 3, (j - 1) * 3:j * 3] = -np.eye(3) * b[j - 1]
    B_a[(j - 1) * 3:j * 3, :3] = np.eye(3)
    C_a[:2, (j - 1) * 3:j * 3] = a[j - 1] * (b[j - 1] * A_3 - A_0)

# State-space matrices for gust aero
for j in range(1, Ng + 1):
    A_g[j - 1, j - 1] = -b_g[j - 1]
    B_g[j - 1, 0] = 1
    C_g[:2, j - 1] = a_g[j - 1] * b_g[j - 1] * A_0[:2, 1]

    # System matrices for the current dynamic pressure
qinf = 1 / 2 * rho * Vinf ** 2

kappa_g = 2 * np.array([[-1, 0], [(x_ea - x_ac) / (c / 2), 2]])
M = np.array([[m_alpha, m_alpha * (x_cg - x_ea) / (c / 2)],
                  [m_alpha * (x_cg - x_ea) / (c / 2), m_alpha * r_alpha ** 2 / (c ** 2 / 4)]])
K = np.array([[k_h, 0], [0, k_a / (c ** 2 / 4)]])

M_ae = (4 * Vinf ** 2 / c ** 2) * M - qinf * kappa_g @ A_2[:, :2]

invM = np.linalg.inv(M_ae)

Cae = - qinf * kappa_g @ (A_1 - (1 / 2) * A_3)
K_ae = - qinf * kappa_g @ A_0
K_ae[:2, :2] += K

# Constructing A_ae, B_ae, C_ae matrices
A_ae = np.block([[np.zeros((3, 3)), np.eye(3), np.zeros((3, 3 * Na))],
                 [-invM @ K_ae, -invM @ Cae, qinf * invM @ kappa_g @ C_a],
                 [np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3 * Na))],
                 [np.zeros((3 * Na, 3)), B_a, A_a]])

B_ae = np.block([[np.zeros((3, 1))],
                 [qinf * invM @ kappa_g @ A_2[:, 2:3]],
                 [1],
                 [np.zeros((3 * Na, 1))]])

C_ae = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

n = A_ae.shape[0]
m = B_ae.shape[1]

V, U = np.linalg.eig(A_ae)



x0 = np.array([0, np.pi / 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Initial state

# Define MPC parameters
N = 20  # Prediction horizon
Q = np.zeros((12, 12))
Q[0, 0] = 0.01
Q[1, 1] = 10
Q[2, 2] = 1
Q[3, 3] = 0.01
Q[4, 4] = 0.01
Q[5, 5] = 0.01

#Q[4, 4] = 100


R = 0.1 * np.eye(1)
R_delta = 0.1 * np.eye(1)
total_steps = 2000

x_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Targeting x_2 = 1

u_min = -1.0
u_max = 1.0
#x_min = np.array([-10, -np.pi / 2, -np.pi / 2, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
#x_max = np.array([10, np.pi / 2, np.pi / 2, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

# Prepare to store simulation data
x_history = np.zeros((12, total_steps + 1))
u_history = np.zeros((1, total_steps))
x_history[:, 0] = x0.flatten()

dT_simulation = 0.01
dT_horizon = 0.01


# Simulation loop
for k in range(total_steps):
    # Define optimization variables
    X = cp.Variable((12, N + 1))
    U = cp.Variable((1, N))

    # Define the cost function
    cost = 0
    for t in range(N):
        cost += cp.quad_form(X[:, t] - np.transpose(x_target), Q)
        cost += cp.quad_form(U[:, t] - U[:, t - 1], R_delta)
        cost += cp.quad_form(U[:, t], R)


    # Define constraints
    constraints = [X[:, 0] == x_history[:, k]]
    for t in range(N):
        constraints += [X[:, t + 1] == (A_ae @ X[:, t] + B_ae @ U[:, t]) * dT_horizon + X[:, t], u_min <= U[:, t], U[:, t] <= u_max]
        #constraints += [x_min <= X[:, t+1], X[:, t+1] <= x_max]


    # Setup and solve the MPC problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    # Apply the first control input
    u_optimal = U.value[:, 0]
    u_history[:, k] = u_optimal.flatten()

    # Update the system state
    x_next = x_history[:, k] + dT_simulation * (A_ae @ x_history[:, k] + B_ae @ u_optimal)
    x_history[:, k + 1] = x_next.flatten()

    print(k)

# Display some results
print("Final state after 2000 steps:", x_history[:, -1])


plt.plot(x_history[1, :] * 180 / np.pi, label='Pitch')
plt.xlabel('Time Step')
plt.ylabel('Pitch')
plt.legend()
plt.show()


plt.plot(x_history[2, :] * 180 / np.pi, label='Flap')
plt.xlabel('Time Step')
plt.ylabel('Flap')
plt.legend()
plt.show()

plt.plot(x_history[0, :], label='Plunge')
plt.xlabel('Time Step')
plt.ylabel('Plunge')
plt.legend()
plt.show()

plt.plot(u_history[0, :] * 180 / np.pi, label='Control input')
plt.xlabel('Time Step')
plt.ylabel('Control input')
plt.legend()
plt.show()
