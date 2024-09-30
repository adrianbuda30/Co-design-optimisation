clear all
load("results/constant_mass_reward_0.2/CartPoleEnv_Tanh_Tsteps_20480000_lr0.0001_hidden_sizes_256_lay2_rewardUpRight_1_obsLen_min1002_constantPoleMass_GaussMix_16.mat")


% Define a matrix of 8 distinct colors
colors = lines(8); 

figure;

for i = 1:8
    % Create a subplot for the current distribution
    subplot(4, 2, i); % 4 rows, 2 columns of subplots

    % Convert cell arrays to matrices for plotting
    mean_vals = combinedData.dist_mean{i}';
    std_vals = combinedData.dist_std{i};
    
    % Create errorbar with distinct color
    h = errorbar(mean_vals, std_vals, '.', 'Color', colors(i, :));
    % Adjust errorbar color to be a lighter shade of the original color
    set(h, 'Color', colors(i,:) + (1-colors(i,:))*0.7); % Mixing the original color with white to lighten it
    set(h, 'MarkerEdgeColor', colors(i, :)); 
    set(h, 'LineWidth', 1.5); % Adjust the line width if necessary

    ylabel('Value');
    titleText = sprintf('Distribution %d - Final mean %.2f - Final std dev %.2f', i, combinedData.dist_mean{i}(end), combinedData.dist_std{i}(end));
    title(titleText);

    % Set the y-axis range from 0 to 10
    ylim([0 10]);
end
for i=1:8
    final_values(i,1) = combinedData.dist_mean{i}(length(combinedData.dist_mean{i}));
    final_values(i,2) = combinedData.dist_std{i}(length(combinedData.dist_std{i}));
end
%%%%%Pole length%%%%%
% Define a matrix of 8 distinct colors (optional)
colors = lines(8);

% Create a figure
figure;

% Loop through each of your pole_length arrays
for j = 1:8
    % Create a subplot for the current pole_length
    subplot(4, 2, j); % 4 rows, 2 columns of subplots
    
    % Plot the current pole_length with its unique color
    plot(combinedData.pole_length{j}, 'Color', colors(j, :));
    ylim([0 10]);

    % Labeling (optional)
    xlabel('Time Step');
    ylabel('Pole Length');
    title(['Pole Length ' num2str(j)]);
end

% Create a figure
figure;

% Loop through each of your pole_length arrays
for j = 1:8
    % Create a subplot for the current pole_length
    subplot(4, 2, j); % 4 rows, 2 columns of subplots
    
    % Plot the current pole_length with its unique color
    plot(combinedData.reward{j}, 'Color', colors(j, :));
    ylim([0 2700]);

    % Labeling (optional)
    xlabel('Time Step');
    ylabel('Pole Length');
    title(['Pole Length ' num2str(j)]);
end