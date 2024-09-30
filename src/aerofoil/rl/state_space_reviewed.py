import numpy as np
import math as m
def state_space(m, x_cg, x_ea, k_h, k_a, k_b):

    chord = 0.254
    thickness = 0.12

    b = chord / 2
    rho = 1.225
    x_ac = 0.25 * chord

    x_cg = x_cg * chord
    x_ea = x_ea * chord


    a = (x_ea - b) / b
    U_inf = 30
    mu = m / (np.pi * rho * b ** 2)
    x_fh = 0.85 * chord
    x_h = 0.80 * chord

    c = (x_h - b) / b
    #k_b = 155
    #k_a / 4


    m_beta = 0.20 * m
    m_alpha = 0.80 * m

    x_alpha = (x_cg - x_ea) / b
    x_beta = (x_fh - x_h) / b


    I_alpha = m_alpha * x_alpha ** 2 * b ** 2 + 0.0449 * chord * 0.8 * (thickness * 0.8 * chord) ** 3
    I_beta = m_beta * x_beta ** 2 * b ** 2 + 0.0449 * chord * 0.2 * (thickness * 0.2 * chord) ** 3


    #I_alpha = 0.0449 * chord * 0.75 * (thickness * 0.75 * chord) ** 3 + m_alpha * (x_cg - x_ea) ** 2 * b ** 2
    #I_beta = m_beta * x_beta ** 2 * b ** 2 + 0.0449 * chord * 0.25 * (thickness * 0.25 * chord) ** 3

    r_alpha = np.sqrt(I_alpha / (m_alpha * b ** 2))
    r_beta = np.sqrt(I_beta / (m_beta * b ** 2))

    x_beta = 0.02


    omega_alpha = np.sqrt(k_a / I_alpha)
    omega_beta = np.sqrt(k_b / I_beta)
    omega_h = np.sqrt(k_h / m_alpha)



    V_inf = U_inf / (omega_alpha * b)

    lambda_1 = 0.014
    lambda_2 = 0.320
    delta_1 = 0.165
    delta_2 = 0.335

    xi_alpha = 0.02
    xi_beta = 0.01
    xi_h = 0.01

    sigma = omega_h / omega_alpha




    T_1 = -1 / 3 * np.sqrt(1 - c ** 2) * (2 + c ** 2) + c * np.arccos(c)
    T_2 = c * (1 - c ** 2) - np.sqrt(1 - c ** 2) * (1 + c ** 2) * np.arccos(c) + c * (np.arccos(c)) ** 2
    T_3 = -(1 / 8 + c ** 2) * (np.arccos(c)) ** 2 + 1 / 4 * c * np.sqrt(1 - c ** 2) * np.arccos(c) * (7 + 2 * c ** 2) - 1 / 8 * (1 - c ** 2) * (5 * c ** 2 + 4)
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

    D_s = 2 * mu * np.block([[sigma * xi_h, 0 , 0],
                             [0 , (r_alpha ** 2) * xi_alpha, 0],
                             [0, 0, (m_beta / m) * (omega_beta / omega_alpha) * (r_beta ** 2) * xi_beta]])

    K_s = mu * np.block([[sigma ** 2, 0, 0],
                         [0 , r_alpha ** 2, 0],
                         [0, 0, (m_beta / m) * ((omega_beta / omega_alpha) ** 2) * (r_beta ** 2)]])

    L_c = mu * np.block([[0],
                         [0],
                         [m_beta / m * (omega_beta / omega_alpha) ** 2 * r_beta ** 2]])

    M_a = np.block([[-1 , a, T_1 / np.pi],
                    [a, -(1 / 8 + a ** 2), -2 * T_13 / np.pi],
                    [T_1 / np.pi, -2 * T_13 / np.pi, T_3 / (np.pi ** 2)]])

    D_a = V_inf * np.block([[-2, -2 * (1 - a), (T_4 - T_11) / np.pi],
                            [1 + 2 * a, a * (1 - 2 * a), 1 / np.pi * (T_8 - T_1 + (c - a) * T_4 + a * T_11)],
                            [-T_12 / np.pi, 1 / np.pi * (2 * T_9 + T_1 + (T_12 - T_4) * (a - 1/2)), T_11 / (2 * np.pi ** 2) * (T_4 - T_12)]])

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

    # Constructing A_ae, B_ae, C_ae  matrices
    A_ae = np.block([[np.zeros((3, 3)), np.block([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.zeros((3, 2))],
                     [-np.linalg.inv(M_s - M_a) @ (K_s - K_a), -np.linalg.inv(M_s - M_a) @ (D_s - D_a), np.linalg.inv(M_s - M_a) @ L_delta],
                     [-Q_a @ np.linalg.inv(M_s - M_a) @ (K_s - K_a) + Q_v, -Q_a @ np.linalg.inv(M_s - M_a) @ (D_s - D_a), Q_a @ np.linalg.inv(M_s - M_a) @ L_delta + L_lambda]])

    B_ae = np.block([[np.linalg.inv(M_s - M_a) @ L_c],
                     [np.zeros((2, 1))],
                     [1],
                     [Q_a @ np.linalg.inv(M_s - M_a) @ L_c]])


    C_ae = np.block([[0, 2 * 3.141 * 0.6, 2 * 3.141 * 0.4, 0, 0, 0, 0, 0]])


    return A_ae, B_ae, C_ae
