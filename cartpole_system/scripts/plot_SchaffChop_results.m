% Define a matrix of 8 distinct colors
colors = lines(1);
windowSize = 500;
figure;
i = 1;
% Convert cell arrays to matrices for plotting
mean_vals = dist_mean;
std_vals = dist_std;

% Create errorbar with distinct color
h = errorbar(mean_vals, std_vals, '.', 'Color', colors(i, :));
% Adjust errorbar color to be a lighter shade of the original color
set(h, 'Color', colors(i,:) + (1-colors(i,:))*0.7); % Mixing the original color with white to lighten it
set(h, 'MarkerEdgeColor', colors(i, :)); 
set(h, 'LineWidth', 1); % Adjust the line width if necessary

ylabel('pole length in m');
xlabel('iterations');
titleText = sprintf('Gauss distribution - Final mean, std dev: %.2f, %.2f', dist_mean(end), dist_std(end));
title(titleText);

% Set the y-axis range from 0 to 10
ylim([0 10]);

%%%%%%%%%%PLOT REWARD%%%%%%%%%%%%%%
figure
% Original data
plot(reward, 'Color', [0.7, 0.7, 0.7]); % Gray color for the original data
hold on;

% Smoothed data using a moving average
smoothed_reward = movmean(reward, windowSize, 'omitnan');
plot(smoothed_reward, '.','Color', colors(i, :), 'LineWidth', 1);
% 
% legend('rewards', 'steps')

% Labeling (optional)
xlabel('iterations');
ylabel('reward');
title('Reward');

ylim([0 5500]);
hold off;
