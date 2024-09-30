import numpy as np
import random
import matplotlib.pyplot as plt
import control as ctrl
from control.matlab import rss, lsim
import scipy.io as sio

from mat4py import loadmat

mat_contents_delta = loadmat('delta.mat')
mat_contents= sio.loadmat('pitch.mat')
pitch = mat_contents['pitch']
delta = mat_contents_delta['delta']


def state_space(m,x_cg,x_ea,k_h,k_a,k_b):

    rho = 1.225
    c = 0.254
    mu = m / (np.pi * rho * c ** 2 / 4)
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
    Vinf = 100
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
    kappa_g = 2 * np.array([[-1, 0], [(x_ea - x_ac) / (c / 2), 2], [(x_fh - x_ea) / (c / 2), 2]])
    M = np.array([[m_alpha, m_alpha * (x_cg - x_ea) / (c / 2), m_beta * (x_fh - x_h) / (c / 2)],
                  [m_alpha * (x_cg - x_ea) / (c / 2), m_alpha * r_alpha ** 2 / (c ** 2 / 4),
                   m_beta * r_beta ** 2 / (c ** 2 / 4) + m_beta * r * r_beta / (c ** 2 / 4)],
                  [m_beta * (x_fh - x_h) / (c / 2),
                   m_alpha * r_alpha ** 2 / (c ** 2 / 4) - m_alpha * r_alpha * r / (c ** 2 / 4),
                   m_beta * r_beta ** 2 / (c ** 2 / 4) + m_beta * r_beta * r_beta / (c ** 2 / 4)]])
    K = np.array([[k_h, 0, 0], [0, k_a / (c ** 2 / 4), k_b / (c ** 2 / 4)], [0, k_a / (c ** 2 / 4), k_b / (c ** 2 / 4)]])

    M_ae = (4 * Vinf ** 2 / c ** 2) * M - qinf * kappa_g @ A_2

    I_alpha = m_alpha * r_alpha ** 2 + 0.0449 * c * 0.8 * (t * 0.8 * c) ** 3
    I_beta = m_beta * r_beta ** 2 + 0.0449 * c * 0.2 * (t * 0.2 * c) ** 3

    omega_h = np.sqrt(k_h / m)
    omega_a = np.sqrt(k_a / I_alpha)
    omega_b = np.sqrt(k_b / I_beta)

    invM = np.linalg.inv(M_ae)

    Cae = - qinf * kappa_g @ (A_1 - (1 / 2) * A_3)
    K_ae = - qinf * kappa_g @ A_0
    K_ae = K_ae + K

    # Constructing A_ae, B_ae, C_ae matrices
    A_ae = np.block([[np.zeros((3, 3)), np.eye(3), np.zeros((3, 3 * Na))],
                     [-invM @ K_ae, -invM @ Cae, qinf * invM @ kappa_g @ C_a],
                     [np.zeros((3 * Na, 3)), B_a, A_a]])

    B_ae = np.block([[np.zeros((3, 1))],
                     [np.zeros((2, 1))],  # [qinf * invM[:2, :2] @ kappa_g[:2, :2] @ A_2[:, 1:2]],
                     [1],
                     [np.zeros((3 * Na, 1))]])

    C_ae = np.block([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    return A_ae, B_ae, C_ae






num_steps = 10000

u_optimal_history = np.zeros((1, num_steps))
u_optimal_history2 = np.zeros((1, num_steps))
u_optimal_history3 = np.zeros((1, num_steps))
dT = 0.01
x_reference = np.array([0])

time = 0

convergence_time = 0
convergence_steps = 0
reward = 0
consecutive_time = 0
checker = 0

tol_ref = 0.0002

#print(x_reference)

m = 1.5
x_cg = 0.6875
x_ea = 0.7109
k_h = 2324.26
k_a = 3977.98
k_b = 390

A_ae,B_ae,C_ae = state_space(m,x_cg,x_ea,k_h,k_a,k_b)

Q1 = np.zeros((12, 12)) # state cost
Q1[0, 0] = 10
Q1[1, 1] = 10
Q1[2, 2] = 1
Q1[3, 3] = 1
Q1[4, 4] = 1
Q1[5, 5] = 1

R = 10 * np.eye(1)  # control cost

K1, S, E = ctrl.lqr(A_ae, B_ae, Q1, R)
best_control_gains1 = K1.reshape((1, 12))

n = A_ae.shape[0]
m = B_ae.shape[1]

V, U = np.linalg.eig(A_ae)

x0 = np.array([0, np.pi / 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
CL0 = C_ae @ x0

Z = np.block([[np.zeros((1, n)), 1]])
N = np.linalg.inv(np.block([[A_ae, B_ae],
[C_ae, 0]])) @ np.transpose(Z)
Nx = N[0:n]
Nu = N[n]


n_bar1 = Nu + K1 @ Nx
n_bar1 = n_bar1[0, 0]

x_history = np.zeros((n, num_steps + 1))
x_history[:, 0] = x0
x_history_control = np.zeros((n, num_steps + 1))
x_history_control[:, 0] = x0
x_history_control2 = np.zeros((n, num_steps + 1))
x_history_control2[:, 0] = x0
x_history_control3 = np.zeros((n, num_steps + 1))
x_history_control3[:, 0] = x0
CL = np.zeros(num_steps + 1)
CL_control = np.zeros(num_steps + 1)
CL[0] = CL0
CL_control[0] = CL0


for k in range(num_steps):
    u_optimal_history[:, k] = - best_control_gains1 @ x_history_control[:, k]
    #mat = np.eye(8) @ np.array([-x_history[0, k] ** 3, -x_history[1, k] ** 3, -x_history[2, k] ** 3, 0, 0, 0, 0, 0])
    dx_next = A_ae @ x_history[:, k]# + mat
    dx_next_control = (A_ae - B_ae @ best_control_gains1) @ x_history_control[:, k] + n_bar1 * B_ae @ x_reference
    x_next = x_history[:, k] + dx_next * dT
    x_next_control = x_history_control[:, k] + dx_next_control * dT
    x_history[:, k + 1] = x_next
    x_history_control[:, k + 1] = x_next_control
    CL[k + 1] = C_ae @ x_history[:, k + 1]
    CL_control[k + 1] = C_ae @ x_history_control[:, k + 1]

    pitch = x_next_control[0,1]
    time += dT
    if abs(pitch - x_reference) < tol_ref:
        consecutive_time += dT
    else:
        consecutive_time = 0

    reward += 1 / (1 + abs(pitch - x_reference) * 180 / np.pi)

    if (consecutive_time > 10 or time >= 100) and checker == 0:
        convergence_time = time
        reward2 = - convergence_time / 100
        checker = 1

    if checker == 1:
        convergence_steps = 1

print(convergence_time)
print(reward)

time = 0

convergence_time = 0
convergence_steps = 0
reward = 0
consecutive_time = 0
checker = 0

tol_ref = 0.0002

m = 1.5
x_cg = 0.6129
x_ea = 0.6193
k_h = 4219.881
k_a = 4713.831
k_b = 390


A_ae,B_ae,C_ae = state_space(m,x_cg,x_ea,k_h,k_a,k_b)

Q2 = np.zeros((12, 12)) # state cost
Q2[0, 0] = 10
Q2[1, 1] = 10
Q2[2, 2] = 1
Q2[3, 3] = 1
Q2[4, 4] = 1
Q2[5, 5] = 1

R = np.eye(1)  # control cost

K2, S, E = ctrl.lqr(A_ae, B_ae, Q2, R)
best_control_gains2 = K2.reshape((1, 12))

n = A_ae.shape[0]
m = B_ae.shape[1]

V, U = np.linalg.eig(A_ae)

x0 = np.array([0, np.pi / 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
CL0 = C_ae @ x0

Z = np.block([[np.zeros((1, n)), 1]])
N = np.linalg.inv(np.block([[A_ae, B_ae],
[C_ae, 0]])) @ np.transpose(Z)
Nx = N[0:n]
Nu = N[n]


n_bar2 = Nu + K2 @ Nx
n_bar2 = n_bar2[0, 0]


for k in range(num_steps):
    u_optimal_history2[:, k] = - best_control_gains2 @ x_history_control2[:, k]
    #mat = np.eye(8) @ np.array([-x_history[0, k] ** 3, -x_history[1, k] ** 3, -x_history[2, k] ** 3, 0, 0, 0, 0, 0])
    dx_next = A_ae @ x_history[:, k]# + mat
    dx_next_control = (A_ae - B_ae @ best_control_gains2) @ x_history_control2[:, k] + n_bar2 * B_ae @ x_reference
    x_next = x_history[:, k] + dx_next * dT
    x_next_control = x_history_control2[:, k] + dx_next_control * dT
    x_history[:, k + 1] = x_next
    x_history_control2[:, k + 1] = x_next_control
    CL[k + 1] = C_ae @ x_history[:, k + 1]
    CL_control[k + 1] = C_ae @ x_history_control2[:, k + 1]

    pitch = x_next_control[0,1]
    time += dT
    if abs(pitch - x_reference) < tol_ref:
        consecutive_time += dT
    else:
        consecutive_time = 0

    reward += 1 / (1 + abs(pitch - x_reference) * 180 / np.pi)

    if (consecutive_time > 10 or time >= 100) and checker == 0:
        convergence_time = time
        reward2 = - convergence_time / 100
        checker = 1

    if checker == 1:
        convergence_steps = 1
print(convergence_time)
print(reward)

time = 0

convergence_time = 0
convergence_steps = 0
reward = 0
consecutive_time = 0
checker = 0

tol_ref = 0.0002

m = 1.5
x_cg = 0.2261
x_ea = 0.3035
k_h = 1870.17
k_a = 3806.65
k_b = 390


A_ae,B_ae,C_ae = state_space(m,x_cg,x_ea,k_h,k_a,k_b)

Q3 = np.zeros((12, 12)) # state cost
Q3[0, 0] = 10
Q3[1, 1] = 10
Q3[2, 2] = 1
Q3[3, 3] = 1
Q3[4, 4] = 1
Q3[5, 5] = 1

R = np.eye(1)  # control cost

K3, S, E = ctrl.lqr(A_ae, B_ae, Q3, R)
best_control_gains3 = K3.reshape((1, 12))

n = A_ae.shape[0]
m = B_ae.shape[1]

V, U = np.linalg.eig(A_ae)

x0 = np.array([0, np.pi / 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
CL0 = C_ae @ x0

Z = np.block([[np.zeros((1, n)), 1]])
N = np.linalg.inv(np.block([[A_ae, B_ae],
[C_ae, 0]])) @ np.transpose(Z)
Nx = N[0:n]
Nu = N[n]


n_bar3 = Nu + K3 @ Nx
n_bar3 = n_bar3[0, 0]



for k in range(num_steps):
    u_optimal_history3[:, k] = - best_control_gains3 @ x_history_control3[:, k]
    #mat = np.eye(8) @ np.array([-x_history[0, k] ** 3, -x_history[1, k] ** 3, -x_history[2, k] ** 3, 0, 0, 0, 0, 0])
    dx_next = A_ae @ x_history[:, k]# + mat
    dx_next_control = (A_ae - B_ae @ best_control_gains3) @ x_history_control3[:, k] + n_bar3 * B_ae @ x_reference
    x_next = x_history[:, k] + dx_next * dT
    x_next_control = x_history_control3[:, k] + dx_next_control * dT
    x_history[:, k + 1] = x_next
    x_history_control3[:, k + 1] = x_next_control
    CL[k + 1] = C_ae @ x_history[:, k + 1]
    CL_control[k + 1] = C_ae @ x_history_control3[:, k + 1]

    pitch = x_next_control[0,1]
    time += dT
    if abs(pitch - x_reference) < tol_ref:
        consecutive_time += dT

    reward += 1 / (1 + abs(pitch - x_reference) * 180 / np.pi)
    if (consecutive_time > 10 or time >= 100) and checker == 0:
        convergence_time = time
        reward2 = - convergence_time / 100
        checker = 1

    if checker == 1:
        convergence_steps = 1

print(convergence_time)
print(reward)



#plt.plot(x_history[0, :], label='Plunge')
#plt.plot(x_history[1, :] * 180 / np.pi, label='Uncontrolled')
plt.plot(x_history_control[1, :] * 180 / np.pi, label='Evolutionary Strategy (R=9678, T=6100)')
plt.plot(x_history_control2[1, :] * 180 / np.pi, label='BayOpt: best reward (R=9714, T=5600)')
plt.plot(x_history_control3[1, :] * 180 / np.pi, label='BayOpt: shortest convergence time (R=9369, T=3900)')
#plt.plot(np.transpose(pitch) * 180 / np.pi, label='RL')

plt.xlabel('Time Step')
plt.ylabel('Pitch incidence (deg)')
plt.grid(color='r', linestyle='-', linewidth=0.2)
plt.legend()
plt.show()

#plt.plot(x_history[0, :], label='Plunge')
#plt.plot(x_history[0, :], label='Uncontrolled')
plt.plot(x_history_control[0, :] * 180 / np.pi, label='Evolutionary Strategy')
plt.plot(x_history_control2[0, :] * 180 / np.pi, label='BayOpt: reward')
plt.plot(x_history_control3[0, :] * 180 / np.pi, label='BayOpt: convergence time')
plt.xlabel('Time Step')
plt.ylabel('Plunge (deg)')
plt.legend()
plt.show()

#plt.plot(x_history[0, :], label='Plunge')
#plt.plot(x_history[2, :] * 180 / np.pi, label='Uncontrolled')
plt.plot(x_history_control[2, :] * 180 / np.pi, label='Evolutionary Strategy (R=9678, T=6100)')
plt.plot(x_history_control2[2, :] * 180 / np.pi, label='BayOpt: best reward (R=9714, T=5600)')
plt.plot(x_history_control3[2, :] * 180 / np.pi, label='BayOpt: shortest convergence time (R=9369, T=3900)')
#plt.plot(np.transpose(delta) * 180 / np.pi, label='RL')
plt.xlabel('Time Step')
plt.ylabel('Flap incidence (deg)')
plt.grid(color='r', linestyle='-', linewidth=0.2)
plt.legend()
plt.show()

plt.plot(u_optimal_history[0, :], label='Control input (Q = 100)')
plt.plot(u_optimal_history2[0, :], label='Control input (Q = 1000)')
plt.plot(u_optimal_history3[0, :], label='Control input (Q = 10000)')
plt.xlabel('Time Step')
plt.ylabel('Control input')
plt.legend()
plt.show()

#plt.plot(CL, label='Uncontrolled')
plt.plot(CL_control, label='Controlled')
plt.xlabel('Time Step')
plt.ylabel('Lift Coefficient')
plt.legend()
plt.show()
