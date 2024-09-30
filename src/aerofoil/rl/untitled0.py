from state_space_LQR import state_space_lqr

m = 1.59;
x_cg = 0.35;
x_ea = 0.45;
k_alpha = 10;
k_h = 10;

Q = state_space_lqr(m, x_cg, x_ea, k_h, k_alpha)
print(Q)

