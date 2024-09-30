clear all
load("results.mat")
figure
subplot(2, 2, 1)
% Plot the pole_vel_simulink data in red with asterisk markers
plot(cart_position_simulink, 'r*')
hold on

% Plot the pole_vel_openAI data in blue with dot markers
plot(cart_position_openAI, 'b.')

% Add labels to the data sets
legend('Simulink', 'OpenAI Gym')

% Add title and axis labels if needed
title('cart position Comparison')
xlabel('Time Step')
ylabel('cart position')
hold off

subplot(2, 2, 2)
% Plot the pole_vel_simulink data in red with asterisk markers
plot(cart_vel_simulink, 'r*')
hold on

% Plot the pole_vel_openAI data in blue with dot markers
plot(cart_vel_openAI, 'b*')

% Add labels to the data sets
legend('Simulink', 'OpenAI Gym')

% Add title and axis labels if needed
title('cart vel Comparison')
xlabel('Time Step')
ylabel('cart vel')
hold off

subplot(2, 2, 3)
% Plot the pole_vel_simulink data in red with asterisk markers
plot(pole_position_simulink, 'r*')
hold on

% Plot the pole_vel_openAI data in blue with dot markers
plot(pole_position_openAI, 'b.')

% Add labels to the data sets
legend('Simulink', 'OpenAI Gym')

% Add title and axis labels if needed
title('pole position Comparison')
xlabel('Time Step')
ylabel('pole position')
hold off

subplot(2, 2, 4)
% Plot the pole_vel_simulink data in red with asterisk markers
plot(pole_vel_simulink, 'r*')
hold on

% Plot the pole_vel_openAI data in blue with dot markers
plot(pole_vel_openAI, 'b.')

% Add labels to the data sets
legend('Simulink', 'OpenAI Gym')

% Add title and axis labels if needed
title('pole vel Comparison')
xlabel('Time Step')
ylabel('pole vel')
hold off