import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math as m

#parameters
S = 0.47            # wing surface area [m^2]
b = 2.59            # wingspan [m]
c = 0.180           # mean chord length [m]
epsilon = 0         # thrust incidence angle [rad] --> NOTE: epsilon is defined from body x in positive pitch (up)
deltamax = 0.3491	# max control surface deflection [rad]

# mass/inertia
mass = 2.65         # total mass of airplane [kg]
Ixx = 0.1512*1.1	# [kg m^2]
Iyy = 0.2785*1.4	# [kg m^2]
Izz = 0.3745*1.4	# [kg m^2]

# environment
rho = 1.225         # air density
g = 9.81            # acceleration of gravity

# aerodynamic/thrust coefficients - component build-up
cD0 = 0.136022235375284
cDa = -0.673702786581529
cDa2 = 5.45456589719510
cL0 = 0.21265754937121
cLa = 10.8060289182568
cLa2 = -46.8323561880705
cLa3 = 60.6017115061355
cLq = 0
cLde = 0
cm0 = 0.0435007528360901
cma = -2.96903143325122
cmq = -106.154115386179
cmde = 6.13078257823941
cT0 = 0
cT1 = 14.7217343655508
cT2 = 0
clb = -0.0154186740460301
clp = -0.164692484392609
clr = 0.0116850531725225
clda = 0.0285
cYb = -0.307330834028566
cnb = 0.0429867867785536
cnp = -0.083852581177017
cnr = -0.082678498998441
cndr = 0.0600000000000000

def forces(state, delta):

    # STATES
    u, v, w, p, q, r = state[0:6]

    # intermediate states (wind-frame)
    VA = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)  # angle of attack
    beta = np.arcsin(v / VA)

    deltaT, deltaE, deltaA_R, deltaA_L, deltaR = delta  # DEFLECTIONS

    # stability-to-body transform
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    H_S2B = np.array([[ca, 0, -sa],
                      [0, 1, 0],
                      [sa, 0, ca]])

    # non-dimensionalized rates
    p_hat = p * b / (2 * 14)
    q_hat = q * c / (2 * 14)
    r_hat = r * b / (2 * 14)
    q_bar_S = 0.5 * rho * VA**2 * S

    # aerodynamic force coefficients (stability-axis)
    # component build-up
    cD = cD0 + cDa * alpha + cDa2 * alpha**2
    cY_s = 0 * cYb * beta
    cL = (cL0 + cLa * alpha + cLa2 * alpha**2 +
          cLa3 * alpha**3 + cLq * q_hat + cLde * deltaE)

    # aerodynamic moment coefficients (stability-axis)
    # component build-up
    cl_s = (clb * beta + clp * p_hat + clr * r_hat +
            clda * deltaA_R - clda * deltaA_L)
    cm_s = (cm0 + cma * alpha + cmq * q_hat +
            cmde * deltaE)
    cn_s = (cnb * beta + cnp * p_hat + cnr * r_hat +
            cndr * deltaR)

    # AERODYNAMIC FORCES (body-frame)
    FA = q_bar_S * np.dot(H_S2B, np.array([-cD, cY_s, -cL]))

    # THRUST FORCE (body-frame)
    FT = (cT0 + cT1 * deltaT + cT2 * deltaT**2) * np.array(
        [np.cos(epsilon), 0, np.sin(epsilon)])

    # AERODYNAMIC MOMENTS (body-frame)
    MA = q_bar_S * np.diag([b, c, b]).dot(np.array([cl_s, cm_s, cn_s]))

    # THRUST MOMENTS (body-frame)
    MT = np.zeros(3)

    F = FA + FT
    M = MA + MT

    return F, M

def deriv(dist, force, moment, state):

    # DISTURBANCE
    wn, we, wd = dist

    # FORCES / MOMENTS
    X, Y, Z = force
    Lm, Mm, Nm = moment

    # STATES
    u, v, w, p, q, r, phi, theta, psi = state[:9]

    # TRANSLATIONAL DYNAMICS

    # u, v, w
    udot = r*v - q*w - g*np.sin(theta) + X/mass
    vdot = p*w - r*u + g*np.sin(phi)*np.cos(theta) + Y/mass
    wdot = q*u - p*v + g*np.cos(phi)*np.cos(theta) + Z/mass

    # n, e, d
    ndot = (u*np.cos(theta)*np.cos(psi) + v*(np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi)) +
            w*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi)) + wn)
    edot = (u*np.cos(theta)*np.sin(psi) + v*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi)) +
            w*(np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)) + we)
    ddot = (-u*np.sin(theta) + v*np.sin(phi)*np.cos(theta) + w*np.cos(phi)*np.cos(theta) + wd)

    # ROTATIONAL DYNAMICS

    # p, q, r
    pdot = (Lm + (Iyy - Izz)*r*q) / Ixx
    qdot = (Mm + (Izz - Ixx)*p*r) / Iyy
    rdot = (Nm + (Ixx - Iyy)*p*q) / Izz

    # phi, theta, psi
    phidot = p + (q*np.sin(phi) + r*np.cos(phi))*np.tan(theta)
    thetadot = q*np.cos(phi) - r*np.sin(phi)
    psidot = (q*np.sin(phi) + r*np.cos(phi)) / np.cos(theta)

    # STATE DIFFERENTIALS
    xdot = np.array([udot, vdot, wdot, pdot, qdot, rdot, phidot, thetadot, psidot, ndot, edot, ddot])

    return xdot


n = 12 # states
m = 5  # inputs
n_target = 2

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
def cost_function(u, x0, coord0, target):

    cost = 0
    dx = np.zeros((n, N + 1))
    x = np.zeros((n, N + 1))
    coord = np.zeros((n_target, N + 1))
    x[:, 0] = x0
    coord[:, 0] = coord0

    for k in range(N):
        cost += np.dot((coord[:, k] - target), np.dot(Q, (coord[:, k] - target))) + np.dot(u[:, k], np.dot(R, u[:, k]))

        force, moment = forces(x_history[:, k], u_optimal_history[:, k])
        dx[:, k] = deriv(wind, force, moment, x_history[:, k])
        x_next = x[:, k] + dx[:, k] * dT_simulation
        x[:, k + 1] = x_next
        coord[:, k + 1] = x[6:7, k + 1]

    return cost



# nonlinear constraint function
def nonlinear_constraint(u):

    return u_min - u[0]

x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
coord0 = np.array([0, 0])

u_init = np.zeros((m, N))
constraints = [{'type': 'ineq', 'fun': nonlinear_constraint}]
bounds = [(u_min, u_max) for _ in range(m * N)]


# simulation
num_steps = 5000
x_history = np.zeros((n, num_steps + 1))
coord = np.zeros((2, num_steps + 1))
x_history[:, 0] = x0
coord[:, 0] = coord0

u_optimal_history = np.zeros((m, num_steps))
counter = 0
target = np.array([2, 2])

wind = np.array([0, 3, 3])

for k in range(num_steps):
    x0 = x_history[:, k]

    result = minimize(
        fun=lambda u: cost_function(u.reshape((m, N)), x0, coord0, target),
        x0=u_init.flatten(),
        method='SLSQP',
       # constraints=constraints,
        bounds=bounds
    )


    # optimal control
    u_optimal = result.x.reshape((m, N))

    u_optimal_history[:, k] = np.array([0, 0, 0, 0, 0])
    if k > 0: #5000:
        u_optimal_history[:, k] = u_optimal[:, 0]

    force, moment = forces(x_history[:, k], u_optimal_history[:, k])
    dx_next =deriv(wind, force, moment, x_history[:, k])
    x_next = x_history[:, k] + dx_next * dT_simulation
    x_history[:, k + 1] = x_next
    coord[:, k + 1] = x_history[6:7, k + 1]

    counter += 1
    print(counter)

plt.plot(x_history[0, :], label='Pitch')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.show()


plt.plot(x_history[1, :], label='Plunge')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.show()

plt.plot(x_history[2, :], label='Plunge')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.show()


plt.plot(x_history[3, :], label='Plunge')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.show()


plt.plot(x_history[4, :], label='Plunge')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.show()


plt.plot(x_history[5, :], label='Plunge')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.show()

plt.plot(u_optimal_history[1, :], label='Control input')
plt.xlabel('Time Step')
plt.ylabel('Control input')
plt.legend()
plt.show()

