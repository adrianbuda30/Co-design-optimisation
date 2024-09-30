import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

# Define the CSV file names for reward and standard deviation
reward_random_csv_file = 'random_design_ESA_2.csv'
reward_hebo_csv_file = 'hebo_gauss_ESA_2.csv'

std_random_csv_file = 'std_random_2.csv'
std_hebo_csv_file = 'std_hebo_2.csv'

# Load the reward and std CSV files into DataFrames
reward_random_df = pd.read_csv(reward_random_csv_file)
reward_hebo_df = pd.read_csv(reward_hebo_csv_file)

std_random_df = pd.read_csv(std_random_csv_file)
std_hebo_df = pd.read_csv(std_hebo_csv_file)

# Concatenate the data, assuming Step 0-999 for random and 1000-1999 for hebo
reward_df = pd.concat([reward_random_df, reward_hebo_df], ignore_index=True)
std_df = pd.concat([std_random_df, std_hebo_df], ignore_index=True)

# Adjust the Step numbering if needed (assuming both start at Step 0)
reward_df['Step'] = list(range(1000)) + list(range(1000, 1569))
std_df['Step'] = list(range(1000)) + list(range(1000, 1569))

reward_df['Step'] = reward_df['Step'] * 1024 * 50
std_df['Step'] = std_df['Step'] * 1024 * 50

# Merge the reward and std DataFrames on 'Step'
merged_df = pd.merge(reward_df, std_df, on='Step', suffixes=('_reward', '_std'))

# Assuming the 'Value' column contains the reward and std values
merged_df['mean_reward'] = merged_df['Value_reward']
merged_df['std_reward'] = merged_df['Value_std']


# Apply a rolling average for smoothing
window_size = 10  # Define the window size for the rolling average
merged_df['smoothed_reward'] = merged_df['mean_reward'].rolling(window=window_size, min_periods=1).mean()
merged_df['smoothed_std'] = merged_df['std_reward'].rolling(window=window_size, min_periods=1).mean()

# Create a new column to distinguish between random_design and hebo_gauss
merged_df['method'] = ['Initial Training' if Step < 1000 * 1024 * 50 else 'HEBO-GMM Optimisation' for Step in merged_df['Step']]

# Plot with Seaborn
plt.figure(figsize=(8, 6))

# Plot the smoothed reward and shaded standard deviation area for each method
sns.lineplot(x='Step', y='smoothed_reward', hue='method', data=merged_df, palette='tab10')

# Fill the area under the curve with different colors for each method
for method in merged_df['method'].unique():
    method_df = merged_df[merged_df['method'] == method]
    plt.fill_between(
        method_df['Step'],
        method_df['smoothed_reward'] - method_df['smoothed_std'] * method_df['smoothed_reward'],
        method_df['smoothed_reward'] + method_df['smoothed_std'] * method_df['smoothed_reward'],
        alpha=0.3
    )
plt.axhline(y=6200, color='black', linestyle='--', label='Reference Reward (Schaff)')
# Turn on the grid and customize it
plt.grid(True, color='gray')
plt.xlabel('No. of Samples', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.show()



# Load the .mat file
mat = scipy.io.loadmat('parameters_2.mat')

# Assuming 'limb_length' is the key in your .mat file containing the data
limb_length = mat['limb_length']

# Extracting the specific columns for torso, thigh, shin, and foot
torso = limb_length[:, 0]
thigh = limb_length[:, 1]
shin = limb_length[:, 2]
foot = limb_length[:, 3]
torsothick = limb_length[:, 7]
thighthick = limb_length[:, 8]
shinthick = limb_length[:, 9]
footthick = limb_length[:, 10]

# Define the window size for smoothing
window_size = 1000

# Calculate moving average, moving max, and moving min
torso_mean = pd.Series(torso).rolling(window=window_size, min_periods=1).mean()
torso_max = pd.Series(torso).rolling(window=window_size, min_periods=1).max()
torso_min = pd.Series(torso).rolling(window=window_size, min_periods=1).min()

thigh_mean = pd.Series(thigh).rolling(window=window_size, min_periods=1).mean()
thigh_max = pd.Series(thigh).rolling(window=window_size, min_periods=1).max()
thigh_min = pd.Series(thigh).rolling(window=window_size, min_periods=1).min()

shin_mean = pd.Series(shin).rolling(window=window_size, min_periods=1).mean()
shin_max = pd.Series(shin).rolling(window=window_size, min_periods=1).max()
shin_min = pd.Series(shin).rolling(window=window_size, min_periods=1).min()

foot_mean = pd.Series(foot).rolling(window=window_size, min_periods=1).mean()
foot_max = pd.Series(foot).rolling(window=window_size, min_periods=1).max()
foot_min = pd.Series(foot).rolling(window=window_size, min_periods=1).min()

torsothick_mean = pd.Series(torsothick).rolling(window=window_size, min_periods=1).mean()
torsothick_max = pd.Series(torsothick).rolling(window=window_size, min_periods=1).max()
torsothick_min = pd.Series(torsothick).rolling(window=window_size, min_periods=1).min()

thighthick_mean = pd.Series(thighthick).rolling(window=window_size, min_periods=1).mean()
thighthick_max = pd.Series(thighthick).rolling(window=window_size, min_periods=1).max()
thighthick_min = pd.Series(thighthick).rolling(window=window_size, min_periods=1).min()

shinthick_mean = pd.Series(shinthick).rolling(window=window_size, min_periods=1).mean()
shinthick_max = pd.Series(shinthick).rolling(window=window_size, min_periods=1).max()
shinthick_min = pd.Series(shinthick).rolling(window=window_size, min_periods=1).min()

footthick_mean = pd.Series(footthick).rolling(window=window_size, min_periods=1).mean()
footthick_max = pd.Series(footthick).rolling(window=window_size, min_periods=1).max()
footthick_min = pd.Series(footthick).rolling(window=window_size, min_periods=1).min()

# Create x-values (steps)
x = np.arange(len(torso)) * 200

# Plotting
plt.figure(figsize=(8, 6))
plt.fill_between(x, torsothick_min, torsothick_max, color='firebrick', alpha=0.2)
plt.fill_between(x, thighthick_min, thighthick_max, color='orange', alpha=0.2)
plt.fill_between(x, shinthick_min, shinthick_max, color='royalblue', alpha=0.2)
plt.fill_between(x, footthick_min, footthick_max, color='forestgreen', alpha=0.2)

plt.plot(x, torsothick_mean, 'firebrick', linewidth=2, label='Torso thickness')
plt.plot(x, thighthick_mean, 'orange', linewidth=2, label='Thigh thickness')
plt.plot(x, shinthick_mean, 'royalblue', linewidth=2, label='Leg thickness')
plt.plot(x, footthick_mean, 'forestgreen', linewidth=2, label='Foot thickness')


# Adjust font size for tick labels
plt.tick_params(axis='both', which='major', labelsize=12)
# Add labels and legend
plt.axhline(y=0.01, color='black', linestyle='--')
plt.axhline(y=0.05, color='black', linestyle='--')
plt.xlabel('No. of Samples',fontsize=12)
plt.ylabel('Parameter Value',fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)
plt.show()
# Plotting
plt.figure(figsize=(8, 6))
plt.fill_between(x, torso_min, torso_max, color='firebrick', alpha=0.2)
plt.fill_between(x, thigh_min, thigh_max, color='orange', alpha=0.2)
plt.fill_between(x, shin_min, shin_max, color='royalblue', alpha=0.2)
plt.fill_between(x, foot_min, foot_max, color='forestgreen', alpha=0.2)

plt.plot(x, torso_mean, 'firebrick', linewidth=2, label='Torso')
plt.plot(x, thigh_mean, 'orange', linewidth=2, label='Thigh')
plt.plot(x, shin_mean, 'royalblue', linewidth=2, label='Leg')
plt.plot(x, foot_mean, 'forestgreen', linewidth=2, label='Foot')

# Adjust font size for tick labels
plt.tick_params(axis='both', which='major', labelsize=12)


# Add labels and legend
plt.xlabel('No. of Samples', fontsize=12)
plt.ylabel('Parameter Value',fontsize=12)
plt.legend(loc='upper right',fontsize=12)
plt.grid(True)
plt.show()