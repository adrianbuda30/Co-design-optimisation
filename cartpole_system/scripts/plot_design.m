function [] = plot_design(pole_length,average_effort)

% Your data
plot(pole_length, average_effort, '*');

% Fit linear regression model
p = polyfit(pole_length, average_effort, 1);

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

end