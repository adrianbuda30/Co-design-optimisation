% Define a matrix of 8 distinct colors
colors = lines(4);
windowSize = 100;
figure
% Convert cell arrays to matrices for plotting
for i=1:4
subplot(2, 2, i); % 4 rows, 2 columns of subplots

mean_vals = arm_length(1,:,i);
% Original data
plot(mean_vals, 'Color', [0.7, 0.7, 0.7]); % Gray color for the original data
hold on;
smoothed_length = movmean(mean_vals, windowSize, 'omitnan');
plot(smoothed_length, '.','Color', colors(i, :), 'LineWidth', 1);
end
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
xlabel('Time Step');
ylabel('reward');
title('Reward');

ylim([0 5500]);
hold off;
