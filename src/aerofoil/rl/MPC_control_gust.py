import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
A_ae = np.block([[np.zeros((3, 3)), np.eye(3), np.zeros((3, 3 * Na)), np.zeros((3, Ng))],
                 [-invM @ K_ae, -invM @ Cae, qinf * invM @ kappa_g @ C_a, qinf * invM @ kappa_g @ C_g],
                 [np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3 * Na)), np.zeros((1, Ng))],
                 [np.zeros((3 * Na, 3)), B_a, A_a, np.zeros((3 * Na, Ng))],
                 [np.zeros((Ng, 3)), np.zeros((Ng, 3)), np.zeros((Ng, 3 * Na)), A_g]])

B_ae = np.block([[np.zeros((3, 1)), np.zeros((3, 1))],
                 [qinf * invM @ kappa_g @ A_2[:, 2:3], np.zeros((2, 1))],
                 [1, 0],
                 [np.zeros((3 * Na, 1)), np.zeros((3 * Na, 1))],
                 [np.zeros((Ng, 1)), B_g]])

C_ae = np.block(
    [[0, 0.8 * CLa0qs, 0.2 * CLd0qs, 0, 0.8 * (CLa1qs + CLa1nc), 0.2 * (CLd1qs + CLd1nc), 0, 0, 0, 0, 0, 0, 0, 0]])

n = A_ae.shape[0]
m = B_ae.shape[1]

V, U = np.linalg.eig(A_ae)

print(V)


# MPC parameters
N = 20 # prediction horizon
Q = np.zeros((14, 14))  # state cost
Q[0, 0] = 10
Q[1, 1] = 10000
Q[2, 2] = 10
Q[3, 3] = 1
Q[4, 4] = 1
Q[5, 5] = 1

#Q[4, 4] = 100


R = 5 * np.eye(1)# input matrix

# Constraints
u_max = 1.0
u_min = -1.0

dT_horizon = 0.01
dT_simulation = 0.01

x_reference = np.zeros((14, 1))
x_reference[1, 0] = np.pi / 90
# cost function
def cost_function(u, x0, CL0, CL_target):

    cost = 0
    dx = np.zeros((n, N + 1))
    x = np.zeros((n, N + 1))
    CL = np.zeros(N + 1)
    x[:, 0] = x0
    CL[0] = CL0

    for k in range(N):
        cost += np.transpose(x[:, k].reshape(-1, 1) - x_reference) @ Q @ (x[:, k].reshape(-1, 1) - x_reference) + u[:, k] @ R @ u[:, k]
        #cost += np.dot(x[:, k], np.dot(Q, x[:, k])) + np.dot(u[:, k], np.dot(R, u[:, k]))

        #mat = np.eye(12) @ np.array([-x[0, k] ** 3, -x[1, k] ** 3, -x[2, k] ** 3, -x[3, k] ** 3, -x[4, k] ** 3, -x[5, k] ** 3, 0, 0, 0, 0, 0, 0])
        dx[:, k] = A_ae @ x[:, k] + B_ae[:, 0:1] @ u[:, k] + B_ae[:, 1:2] @ np.array([W_g]) # + mat
        x[:, k + 1] = x[:, k] + dx[:, k] * dT_horizon
        CL[k + 1] = C_ae @ x[:, k + 1]

    return cost



# nonlinear constraint function
def nonlinear_constraint(u):

    return u_min - u[0]

x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
CL0 = C_ae @ x0

u_init = np.zeros((m - 1, N))
constraints = [{'type': 'ineq', 'fun': nonlinear_constraint}]
bounds = [(u_min, u_max) for _ in range((m - 1) * N)]


# simulation
num_steps = 5000
x_history = np.zeros((n, num_steps + 1))
control = np.zeros(num_steps)
x_history[:, 0] = x0
CL = np.zeros(num_steps + 1)
CL[0] = CL0

u_optimal_history = np.zeros((m - 1, num_steps))
counter = 0
CL_target = 0.0

for k in range(num_steps):
    x0 = x_history[:, k]
    CL0 = C_ae @ x0

    #if k % 3000 == 0:
    #    CL_target = -CL_target

    result = minimize(
        fun=lambda u: cost_function(u.reshape((m - 1, N)), x0, CL0, CL_target),
        x0=u_init.flatten(),
        method='SLSQP',
       # constraints=constraints,
        bounds=bounds
    )


    # optimal control
    u_optimal = result.x.reshape((m - 1, N))

    u_optimal_history[:, k] = 0
    if k > 0:
        u_optimal_history[:, k] = u_optimal[:, 0]
    #mat = np.eye(12) @ np.array([-x_history[0, k] ** 3, -x_history[1, k] ** 3, -x_history[2, k] ** 3, -x_history[3, k] ** 3, -x_history[4, k] ** 3, -x_history[5, k] ** 3, 0, 0, 0, 0, 0, 0])
    dx_next = A_ae @ x_history[:, k] + B_ae[:, 0:1] @ u_optimal_history[:, k] + B_ae[:, 1:2] @ np.array([W_g]) #+ mat
    control[k] = u_optimal_history[0, k]
    x_next = x_history[:, k] + dx_next * dT_simulation
    x_history[:, k + 1] = x_next
    CL[k + 1] = C_ae @ x_history[:, k + 1]

    counter += 1
    print(counter)


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

plt.plot(control[:] * 180 / np.pi, label='Control input')
plt.xlabel('Time Step')
plt.ylabel('Control input')
plt.legend()
plt.show()



plt.plot(CL, label='Lift coefficient')
plt.xlabel('Time Step')
plt.ylabel('Lift Coefficient')
plt.legend()
plt.show()