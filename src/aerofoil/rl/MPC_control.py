import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math as m

#inputs

m = 1
x_cg = 0.62
x_ea = 0.2613
k_h = 1800
k_a = 14500

chord = 0.254

b = chord / 2
rho = 1.225
x_ac = 0.25 * chord

x_cg = x_cg * chord
x_ea = x_ea * chord

a = (x_ea - b) / b
U_inf = 10
mu = m / (np.pi * rho * b ** 2)
f = 0.428
x_fh = 0.86 * chord
x_h = 0.8 * chord

c = (x_h - b) / b
k_b = 155 # k_a / 4

r_alpha = abs(x_cg - x_ea) / b
r_beta = abs(x_fh - x_h) / b

m_beta = 0.2 * m

I_alpha = m * r_alpha ** 2 * b ** 2
I_beta = m_beta * r_beta ** 2 * b ** 2

omega_alpha = np.sqrt(k_a / I_alpha)
omega_beta = np.sqrt(k_b / I_beta)
omega_h = np.sqrt(k_h / m)

V_inf = U_inf / (omega_alpha * b)

lambda_1 = 0.045
lambda_2 = 0.3
delta_1 = 0.165
delta_2 = 0.335

xi_alpha = 0.02
xi_beta = 0.01
xi_h = 0.01

sigma = omega_h / omega_alpha

x_alpha = (x_cg - x_ea) / b
x_beta = 0.02

# Theodorsen's coefficients

T_1 = -1 / 3 * np.sqrt(1 - c ** 2) * (2 + c ** 2) + c * np.arccos(c)
T_2 = c * (1 - c ** 2) - np.sqrt(1 - c ** 2) * (1 + c ** 2) * np.arccos(c) + c * (np.arccos(c)) ** 2
T_3 = -(1 / 8 + c ** 2) * (np.arccos(c)) ** 2 + 1 / 4 * c * np.sqrt(1 - c ** 2) * (7 + 2 * c ** 2) - 1 / 8 * (
            1 - c ** 2) * (5 * c ** 2 + 4)
T_4 = -np.arccos(c) + c * np.sqrt(1 - c ** 2)
T_5 = -(1 - c ** 2) - (np.arccos(c)) ** 2 + 2 * c * np.sqrt(1 - c ** 2) * np.arccos(c)
T_6 = T_2
T_7 = -(1 / 8 + c ** 2) * np.arccos(c) + 1 / 8 * c * np.sqrt(1 - c ** 2) * (7 + 2 * c ** 2)
T_8 = -1 / 3 * np.sqrt(1 - c ** 2) * (2 * c ** 2 + 1) + c * np.arccos(c)
T_9 = 1 / 2 * (1 / 3 * (1 - c ** 2) ** (3 / 2) + a * T_4)
T_10 = np.sqrt(1 - c ** 2) + np.arccos(c)
T_11 = np.arccos(c) * (1 - 2 * c) + np.sqrt(1 - c ** 2) * (2 - c)
T_12 = np.sqrt(1 - c ** 2) * (2 + c) - np.arccos(c) * (1 + 2 * c)
T_13 = 1 / 2 * (-T_7 - (c - a) * T_1)
T_14 = 1 / 16 + 1 / 2 * a * c

M_s = mu * np.block([[1, x_alpha, m_beta / m * x_beta],
                     [x_alpha, r_alpha ** 2, m_beta / m * ((c - a) * x_beta + r_beta ** 2)],
                     [m_beta / m * x_beta, m_beta / m * ((c - a) * x_beta + r_beta ** 2), m_beta / m * (r_beta ** 2)]])

D_s = 2 * mu * np.block([[sigma * xi_h, 0, 0],
                         [0, (r_alpha ** 2) * xi_alpha, 0],
                         [0, 0, (m_beta / m) * (omega_beta / omega_alpha) * (r_beta ** 2) * xi_beta]])

K_s = mu * np.block([[sigma ** 2, 0, 0],
                     [0, r_alpha ** 2, 0],
                     [0, 0, (m_beta / m) * ((omega_beta / omega_alpha) ** 2) * (r_beta ** 2)]])

L_c = mu * np.block([[0],
                     [0],
                     [m_beta / m * (omega_beta / omega_alpha) ** 2 * r_beta ** 2]])

M_a = np.block([[-1, a, T_1 / np.pi],
                [a, -(1 / 8 + a ** 2), -2 * T_13 / np.pi],
                [T_1 / np.pi, -2 * T_13 / np.pi, T_3 / (np.pi ** 2)]])

D_a = V_inf * np.block([[-2, -2 * (1 - a), (T_4 - T_11) / np.pi],
                        [1 + 2 * a, a * (1 - 2 * a), 1 / np.pi * (T_8 - T_1 + (c - a) * T_4 + a * T_11)],
                        [-T_12 / np.pi, 1 / np.pi * (2 * T_9 + T_1 + (T_12 - T_4) * (a - 1 / 2)),
                         T_11 / (2 * np.pi ** 2) * (T_4 - T_12)]])

K_a = V_inf ** 2 * np.block([[0, -2, -2 * T_10 / np.pi],
                             [0, 1 + 2 * a, 1 / np.pi * (2 * a * T_10 - T_4)],
                             [0, -T_12 / np.pi, -1 / (np.pi ** 2) * (T_5 - T_10 * (T_4 - T_12))]])

L_delta = 2 * V_inf * np.block([[delta_1, delta_2],
                                [-(1 / 2 + a) * delta_1, -(1 / 2 + a) * delta_2],
                                [T_12 * delta_1 / (2 * np.pi), T_12 * delta_2 / (2 * np.pi)]])

Q_a = np.block([[1, 1 / 2 - a, T_11 / (2 * np.pi)],
                [1, 1 / 2 - a, T_11 / (2 * np.pi)]])

Q_v = U_inf * np.block([[0, 1, T_10 / np.pi],
                        [0, 1, T_10 / np.pi]])

L_lambda = V_inf * np.block([[-lambda_1, 0],
                             [0, -lambda_2]])

A_ae = np.block([[np.zeros((3, 3)), np.block([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.zeros((3, 2))],
                 [-np.linalg.inv(M_s - M_a) @ (K_s - K_a), -np.linalg.inv(M_s - M_a) @ (D_s - D_a),
                  np.linalg.inv(M_s - M_a) @ L_delta],
                 [-Q_a @ np.linalg.inv(M_s - M_a) @ (K_s - K_a) + Q_v, -Q_a @ np.linalg.inv(M_s - M_a) @ (D_s - D_a),
                  Q_a @ np.linalg.inv(M_s - M_a) @ L_delta + L_lambda]])

B_ae = np.block([[np.zeros((5, 1))],
                 [1],
                 [np.zeros((2, 1))]])

C_ae = np.block([[0, 2 * np.pi * 0.7, 2 * np.pi * 0.3, 0, 0, 0, 0, 0]])


n = A_ae.shape[0]  # states
m = B_ae.shape[1]  # inputs

# MPC parameters
N = 20 # prediction horizon
Q = 1000 #np.block([[100, 0, 0, 0, 0, 0, 0, 0],
         #     [0, 10, 0, 0, 0, 0, 0, 0],
         #     [0, 0, 25, 0, 0, 0, 0, 0],
         #     [0, 0, 0, 100, 0, 0, 0, 0],
         #     [0, 0, 0, 0, 500, 0, 0, 0],
         #     [0, 0, 0, 0, 0, 10, 0, 0],
         #     [0, 0, 0, 0, 0, 0, 25, 0],
         #     [0, 0, 0, 0, 0, 0, 0, 10]])  # state matrix
R = 10 # input matrix

# Constraints
u_max = 10.0
u_min = -10.0

dT_horizon = 0.05
dT_simulation = 0.01
# cost function
def cost_function(u, x0, CL0, CL_target):

    cost = 0
    dx = np.zeros((n, N + 1))
    x = np.zeros((n, N + 1))
    CL = np.zeros(N + 1)
    x[:, 0] = x0
    CL[0] = CL0

    for k in range(N):
        cost += np.dot((CL[k] - CL_target), np.dot(Q, (CL[k] - CL_target))) + np.dot(u[:, k], np.dot(R, u[:, k]))

        mat = np.eye(8) @ np.array([-x[0, k] ** 3, -x[1, k] ** 3, -x[2, k] ** 3, 0, 0, 0, 0, 0])
        dx[:, k] = A_ae @ x[:, k] + B_ae @ u[:, k] + mat
        print(u)
        x[:, k + 1] = x[:, k] + dx[:, k] * dT_horizon
        CL[k + 1] = C_ae @ x[:, k + 1]

    return cost



# nonlinear constraint function
def nonlinear_constraint(u):

    return u_min - u[0]

x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
CL0 = C_ae @ x0

u_init = np.zeros((m, N))
constraints = [{'type': 'ineq', 'fun': nonlinear_constraint}]
bounds = [(u_min, u_max) for _ in range(m * N)]


# simulation
num_steps = 5000
x_history = np.zeros((n, num_steps + 1))
control = np.zeros(num_steps)
x_history[:, 0] = x0
CL = np.zeros(num_steps + 1)
CL[0] = CL0

u_optimal_history = np.zeros((m, num_steps))
counter = 0
CL_target = 0.2

for k in range(num_steps):
    x0 = x_history[:, k]
    CL0 = C_ae @ x0

    #if k % 3000 == 0:
    #    CL_target = -CL_target

    result = minimize(
        fun=lambda u: cost_function(u.reshape((m, N)), x0, CL0, CL_target),
        x0=u_init.flatten(),
        method='SLSQP',
       # constraints=constraints,
        bounds=bounds
    )


    # optimal control
    u_optimal = result.x.reshape((m, N))

    u_optimal_history[:, k] = 0
    if k > 0: #5000:
        u_optimal_history[:, k] = u_optimal[:, 0]
    mat = np.eye(8) @ np.array([-x_history[0, k] ** 3, -x_history[1, k] ** 3, -x_history[2, k] ** 3, 0, 0, 0, 0, 0])
    dx_next = A_ae @ x_history[:, k] + B_ae @ u_optimal_history[:, k] + mat
    control[k] = dx_next[5]
    x_next = x_history[:, k] + dx_next * dT_simulation
    x_history[:, k + 1] = x_next
    CL[k + 1] = C_ae @ x_history[:, k + 1]

    counter += 1
    print(counter)


plt.plot(x_history[1, :] * 180 / np.pi, label='Pitch')
plt.plot(x_history[2, :] * 180 / np.pi, label='Flap')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.show()

plt.plot(x_history[0, :], label='Plunge')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.show()

plt.plot(control[:], label='Control input')
plt.xlabel('Time Step')
plt.ylabel('Control input')
plt.legend()
plt.show()

plt.plot(CL, label='Lift coefficient')
plt.xlabel('Time Step')
plt.ylabel('Lift Coefficient')
plt.legend()
plt.show()