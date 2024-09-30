% Define a matrix of 8 distinct colors
colors = lines(8);
windowSize = 200;
figure;

for i = 1:4
    
    % Create a subplot for the current distribution
    subplot(4, 4, 2*i-1); % 4 rows, 2 columns of subplots

    % Convert cell arrays to matrices for plotting
    mean_vals = dist_mean{i}';
    std_vals = dist_std{i};
    
    % Create errorbar with distinct color
    h = errorbar(mean_vals, std_vals, '.', 'Color', colors(i, :));
    % Adjust errorbar color to be a lighter shade of the original color
    set(h, 'Color', colors(i,:) + (1-colors(i,:))*0.7); % Mixing the original color with white to lighten it
    set(h, 'MarkerEdgeColor', colors(i, :)); 
    set(h, 'LineWidth', 1); % Adjust the line width if necessary

    ylabel('Value');
    titleText = sprintf('Distribution %d - Final mean, std dev: %.2f, %.2f', i, dist_mean{i}(end), dist_std{i}(end));
    title(titleText);

    % Set the y-axis range from 0 to 10
    ylim([0 10]);

    %%%%%%%%%%PLOT REWARD%%%%%%%%%%%%%%
    subplot(4, 4, 2*i); % 4 rows, 2 columns of subplots

    % Original data
    plot(reward{i}, 'Color', [0.7, 0.7, 0.7]); % Gray color for the original data
    hold on;

    % Smoothed data using a moving average
    smoothed_reward = movmean(reward{i}, windowSize, 'omitnan');
    plot(smoothed_reward, '.','Color', colors(i, :), 'LineWidth', 1);
    % 
    % legend('rewards', 'steps')

    % Labeling (optional)
    xlabel('Time Step');
    ylabel('reward');
    title('Reward');
    
    ylim([0 5500]);
    hold off;
end

% %%%%%Total steps%%%%%

% Define a matrix of 8 distinct colors
colors = lines(8);
windowSize = 200;
figure;

for i = 1:8
    
    % Create a subplot for the current distribution
    subplot(4, 4, 2*i-1); % 4 rows, 2 columns of subplots

    % Convert cell arrays to matrices for plotting
    mean_vals = dist_mean{i}';
    std_vals = dist_std{i};
    
    % Create errorbar with distinct color
    h = errorbar(mean_vals, std_vals, '.', 'Color', colors(i, :));
    % Adjust errorbar color to be a lighter shade of the original color
    set(h, 'Color', colors(i,:) + (1-colors(i,:))*0.7); % Mixing the original color with white to lighten it
    set(h, 'MarkerEdgeColor', colors(i, :)); 
    set(h, 'LineWidth', 1); % Adjust the line width if necessary

    ylabel('Value');
    titleText = sprintf('Distribution %d - Final mean, std dev: %.2f, %.2f', i, dist_mean{i}(end), dist_std{i}(end));
    title(titleText);

    % Set the y-axis range from 0 to 10
    ylim([0 10]);

    %%%%%%%%%%PLOT REWARD%%%%%%%%%%%%%%
    subplot(4, 4, 2*i); % 4 rows, 2 columns of subplots

    % Original data
    plot(iteration{i}, 'Color', [0.7, 0.7, 0.7]); % Gray color for the original data
    hold on;

    % Smoothed data using a moving average
    smoothed_iteration = movmean(iteration{i}, windowSize, 'omitnan');
    plot(smoothed_iteration, '.','Color', colors(i, :), 'LineWidth', 1);
    % 
    % legend('rewards', 'steps')

    % Labeling (optional)
    xlabel('Time Step');
    ylabel('steps');
    title('Step');
    
    ylim([0 5500]);
    hold off;
end


% Convert cell arrays to numeric arrays
pole_length_numeric = cell2mat(pole_length);
reward_numeric = cell2mat(reward);

figure;  % Create a new figure window

% Determine bin edges
BinWidth_pole_length = 0.2;
binEdges = min(pole_length_numeric):BinWidth_pole_length:max(pole_length_numeric);

% Initialize arrays to hold bin centers and average rewards
binCenters = (binEdges(1:end-1) + binEdges(2:end))/2;
averageRewards = zeros(1,length(binCenters));

% Loop through each bin and calculate average reward
for i = 1:length(binCenters)
    % Logical indexing to find pole_length values within current bin
    inBin = pole_length_numeric >= binEdges(i) & pole_length_numeric < binEdges(i+1);
    % Calculate average reward for current bin
    averageRewards(i) = mean(reward_numeric(inBin));
end

% Plot average reward for each bin
plot(binCenters, averageRewards, 'o-');  % 'o-' creates a line plot with circle markers
xlabel('Pole Length');  % Label x-axis
ylabel('Average Reward');  % Label y-axis
title('Average Reward vs Pole Length');  % Set title
% ylim([3000 5500]);  % Set y-axis limits
% xlim([0 3]);  % Set x-axis limits

