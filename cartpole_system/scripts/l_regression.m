load("EVAL_CartPoleEnv_Tanh_Tsteps_122880000_lr0.0001_hidden_sizes_256_lay2_rewardUpRight_1_obsLen_min1002_polemass_GaussMix_10.mat")

% Your data
plot(pole_length, steps, '*');

% Fit linear regression model
p = polyfit(pole_length, steps, 1);

% Evaluate the polynomial to get the y-values of the regression line
y_fit = polyval(p,pole_length);

% Plot the regression line
hold on;  % Keeps the previous plot
plot(pole_length, y_fit, '-r');  % '-r' makes the line red

xlabel('Pole Length');
ylabel('Average Effort');
title('Linear Regression of Average Effort vs Pole Length');
legend('Data', 'Linear fit');
hold off;  % Releases the plot
