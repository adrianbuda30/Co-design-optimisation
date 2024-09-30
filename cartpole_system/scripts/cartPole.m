clear;
clc;

% Constants
g = -9.81; % acceleration due to gravity
m_cart = 1; % mass of the cart
m_pole = 0.1; % mass of the pole
l = 1; % length of the pole
dt = 0.02; % time step
t_final = 100; % final time
n_steps = t_final / dt; % number of time steps

% Initial state [x, x_dot, theta, theta_dot]
state = [0, 0, pi/4, 0];

% Initialize plot
figure;
axis([-2, 2, -1, 2]);
axis equal;
grid on;

for i = 1:n_steps
    % Extract state variables
    x = state(1);
    x_dot = state(2);
    theta = state(3);
    theta_dot = state(4);

    % Equations of motion (Euler method)
    x_ddot = (m_pole * sin(theta) * (l * theta_dot^2 + g * cos(theta))) / (m_cart + m_pole * sin(theta)^2);
    theta_ddot = (-x_ddot * cos(theta) - g * sin(theta)) / l;

    % Update state
    state(1) = x + x_dot * dt;
    state(2) = x_dot + x_ddot * dt;
    state(3) = theta + theta_dot * dt;
    state(4) = theta_dot + theta_ddot * dt;

    % Plotting
    cla; % clear previous plot
    rectangle('Position', [x - 0.1, -0.1, 0.2, 0.2], 'EdgeColor', 'b');
    line([x, x + l * sin(theta)], [0, l * cos(theta)], 'Color', 'r', 'LineWidth', 2);
    drawnow;

    pause(dt);
end
