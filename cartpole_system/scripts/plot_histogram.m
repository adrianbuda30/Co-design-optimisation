% Assuming pole_length and reward are pre-defined arrays
figure;  % Create a new figure window
BinWidth_pole_length = 0.2;
binEdges = min(pole_length):BinWidth_pole_length:max(pole_length);  % Determine bin edges

% Initialize arrays to hold bin centers and average rewards
binCenters = (binEdges(1:end-1) + binEdges(2:end))/2;
averageRewards = zeros(1,length(binCenters));

% Loop through each bin and calculate average reward
for i = 1:length(binCenters)
    % Logical indexing to find pole_length values within current bin
    inBin = pole_length >= binEdges(i) & pole_length < binEdges(i+1);
    % Calculate average reward for current bin
    averageRewards(i) = mean(reward(inBin));
end

% Plot average reward for each bin
plot(binCenters, averageRewards, 'o-');  % 'o-' creates a line plot with circle markers
xlabel('Pole Length');  % Label x-axis
ylabel('Average Reward');  % Label y-axis
title('Average Reward vs Pole Length');  % Set title
