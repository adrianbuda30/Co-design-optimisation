figure
idx = 1;
for i = 1:4:length(reward)
avg_reward(idx) = sum(reward(i:i+3))/4;
avg_pole_length(idx) = sum(pole_length(i:i+3))/4;
idx = idx + 1;
end

plot(avg_pole_length,avg_reward, '*')


% Assuming pole_length and reward are pre-defined arrays
figure;  % Create a new figure window
disp_start = 1;
disp_end = length(avg_pole_length);
% Scatter plot for reward against pole_length
plot(avg_pole_length(disp_start:disp_end), avg_reward(disp_start:disp_end), '*');  % Create scatter plot
xlabel('Pole Length');  % Label x-axis
ylabel('avg_reward');  % Label y-axis
title('avg_reward vs Pole Length');  % Set title


% Assuming pole_length and avg_reward are pre-defined arrays
figure;  % Create a new figure window
BinWidth_pole_length = 0.5;
binEdges = min(avg_pole_length):BinWidth_pole_length:max(avg_pole_length);  % Determine bin edges

% Initialize arrays to hold bin centers and average rewards
binCenters = (binEdges(1:end-1) + binEdges(2:end))/2;
averageRewards = zeros(1,length(binCenters));

% Loop through each bin and calculate average avg_reward
for i = 1:length(binCenters)
    % Logical indexing to find pole_length values within current bin
    inBin = avg_pole_length >= binEdges(i) & avg_pole_length < binEdges(i+1);
    % Calculate average avg_reward for current bin
    averageRewards(i) = mean(avg_reward(inBin));
end

% Plot average avg_reward for each bin
plot(binCenters, averageRewards, 'o-');  % 'o-' creates a line plot with circle markers
xlabel('Pole Length');  % Label x-axis
ylabel('Average avg_reward');  % Label y-axis
title('Average avg_reward vs Pole Length');  % Set title
clear all

