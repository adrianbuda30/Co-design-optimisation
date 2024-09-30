% Load the variables from the first file into a structure
data1 = load("results/constant_mass_reward_0.2/CartPoleEnv_Tanh_Tsteps_40960000_lr0.0001_hidden_sizes_256_lay2_rewardUpRight_1_obsLen_min1002_constantPoleMass_GaussMix_10.mat");

% Load the variables from the second file into a different structure
data2 = load("results/constant_mass_reward_0.2/CartPoleEnv_Tanh_Tsteps_20480000_lr0.0001_hidden_sizes_256_lay2_rewardUpRight_1_obsLen_min1002_constantPoleMass_GaussMix_15.mat");
for s=1:8
data1.dist_std{1,s} = data1.dist_std{1,s}';
data2.dist_std{1,s} = data2.dist_std{1,s}';
end
% Initialize a structure to hold the combined data
combinedData = data1;

% Iterate through each field and append
fieldNames = fieldnames(data1);
for i = 1:length(fieldNames)
    fieldName = fieldNames{i};
    for k = 1:8 % Or whatever the length of your cell array is
        % Debugging: Check sizes
        disp(['Size of data1 for field ', fieldName, ' cell ', num2str(k), ':']);
        disp(size(data1.(fieldName){k}));
        
        disp(['Size of data2 for field ', fieldName, ' cell ', num2str(k), ':']);
        disp(size(data2.(fieldName){k}));

        % Concatenate
        combinedData.(fieldName){k} = [data1.(fieldName){k} data2.(fieldName){k}];
    end
end

% Save the combined data
save("combinedFile.mat", '-struct', 'combinedData');

