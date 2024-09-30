import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

# Define the CSV file names for reward and standard deviation
reward_random_csv_file = 'random_design_ESA.csv'
reward_hebo_csv_file = 'hebo_gauss_ESA.csv'

std_random_csv_file = 'std_random.csv'
std_hebo_csv_file = 'std_hebo.csv'

# Load the reward and std CSV files into DataFrames
reward_random_df = pd.read_csv(reward_random_csv_file)
reward_hebo_df = pd.read_csv(reward_hebo_csv_file)

std_random_df = pd.read_csv(std_random_csv_file)
std_hebo_df = pd.read_csv(std_hebo_csv_file)

# Concatenate the data, assuming Step 0-999 for random and 1000-1999 for hebo
reward_df = pd.concat([reward_random_df, reward_hebo_df], ignore_index=True)
std_df = pd.concat([std_random_df, std_hebo_df], ignore_index=True)

# Adjust the Step numbering if needed (assuming both start at Step 0)
reward_df['Step'] = list(range(336)) + list(range(336, 770))
std_df['Step'] = list(range(336)) + list(range(336, 769))

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
merged_df['method'] = ['Initial Training' if Step < 336 * 1024 * 50 else 'HEBO-GMM Optimisation' for Step in merged_df['Step']]

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
plt.axhline(y=6600, color='black', linestyle='--', label='Reference Reward (Schaff)')
# Turn on the grid and customize it
plt.grid(True, color='gray')
plt.xlabel('No. of Samples', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.show()

# Load the .mat file
mat = scipy.io.loadmat('parameters.mat')

# Assuming 'limb_length' is the key in your .mat file containing the data
limb_length = mat['limb_length']

# Extracting the specific columns for torso, thigh, Leg, and foot
body = limb_length[:, 0]

thigh_left = limb_length[:, 1]
shin_left = limb_length[:, 9]
foot_left = limb_length[:, 17]

thighthick_left = limb_length[:, 5]
shinthick_left = limb_length[:, 13]
footthick_left = limb_length[:, 21]

thigh_right= limb_length[:, 2]
shin_right = limb_length[:, 10]
foot_right = limb_length[:, 18]

thighthick_right = limb_length[:, 6]
shinthick_right = limb_length[:, 14]
footthick_right = limb_length[:, 22]

thigh_front = limb_length[:, 4]
shin_front = limb_length[:, 12]
foot_front = limb_length[:, 20]

thighthick_front = limb_length[:, 8]
shinthick_front = limb_length[:, 16]
footthick_front = limb_length[:, 24]

thigh_back = limb_length[:, 3]
shin_back = limb_length[:, 11]
foot_back = limb_length[:, 19]

thighthick_back = limb_length[:, 7]
shinthick_back = limb_length[:, 15]
footthick_back = limb_length[:, 23]

# Define the window size for smoothing
window_size = 1000

# Calculate moving average, moving max, and moving min
body_mean = pd.Series(body).rolling(window=window_size, min_periods=1).mean()
body_max = pd.Series(body).rolling(window=window_size, min_periods=1).max()
body_min = pd.Series(body).rolling(window=window_size, min_periods=1).min()

thigh_left_mean = pd.Series(thigh_left).rolling(window=window_size, min_periods=1).mean()
thigh_left_max = pd.Series(thigh_left).rolling(window=window_size, min_periods=1).max()
thigh_left_min = pd.Series(thigh_left).rolling(window=window_size, min_periods=1).min()

shin_left_mean = pd.Series(shin_left).rolling(window=window_size, min_periods=1).mean()
shin_left_max = pd.Series(shin_left).rolling(window=window_size, min_periods=1).max()
shin_left_min = pd.Series(shin_left).rolling(window=window_size, min_periods=1).min()

foot_left_mean = pd.Series(foot_left).rolling(window=window_size, min_periods=1).mean()
foot_left_max = pd.Series(foot_left).rolling(window=window_size, min_periods=1).max()
foot_left_min = pd.Series(foot_left).rolling(window=window_size, min_periods=1).min()

thigh_right_mean = pd.Series(thigh_right).rolling(window=window_size, min_periods=1).mean()
thigh_right_max = pd.Series(thigh_right).rolling(window=window_size, min_periods=1).max()
thigh_right_min = pd.Series(thigh_right).rolling(window=window_size, min_periods=1).min()

shin_right_mean = pd.Series(shin_right).rolling(window=window_size, min_periods=1).mean()
shin_right_max = pd.Series(shin_right).rolling(window=window_size, min_periods=1).max()
shin_right_min = pd.Series(shin_right).rolling(window=window_size, min_periods=1).min()

foot_right_mean = pd.Series(foot_right).rolling(window=window_size, min_periods=1).mean()
foot_right_max = pd.Series(foot_right).rolling(window=window_size, min_periods=1).max()
foot_right_min = pd.Series(foot_right).rolling(window=window_size, min_periods=1).min()

thigh_back_mean = pd.Series(thigh_back).rolling(window=window_size, min_periods=1).mean()
thigh_back_max = pd.Series(thigh_back).rolling(window=window_size, min_periods=1).max()
thigh_back_min = pd.Series(thigh_back).rolling(window=window_size, min_periods=1).min()

shin_back_mean = pd.Series(shin_back).rolling(window=window_size, min_periods=1).mean()
shin_back_max = pd.Series(shin_back).rolling(window=window_size, min_periods=1).max()
shin_back_min = pd.Series(shin_back).rolling(window=window_size, min_periods=1).min()

foot_back_mean = pd.Series(foot_back).rolling(window=window_size, min_periods=1).mean()
foot_back_max = pd.Series(foot_back).rolling(window=window_size, min_periods=1).max()
foot_back_min = pd.Series(foot_back).rolling(window=window_size, min_periods=1).min()

thigh_front_mean = pd.Series(thigh_front).rolling(window=window_size, min_periods=1).mean()
thigh_front_max = pd.Series(thigh_front).rolling(window=window_size, min_periods=1).max()
thigh_front_min = pd.Series(thigh_front).rolling(window=window_size, min_periods=1).min()

shin_front_mean = pd.Series(shin_front).rolling(window=window_size, min_periods=1).mean()
shin_front_max = pd.Series(shin_front).rolling(window=window_size, min_periods=1).max()
shin_front_min = pd.Series(shin_front).rolling(window=window_size, min_periods=1).min()

foot_front_mean = pd.Series(foot_front).rolling(window=window_size, min_periods=1).mean()
foot_front_max = pd.Series(foot_front).rolling(window=window_size, min_periods=1).max()
foot_front_min = pd.Series(foot_front).rolling(window=window_size, min_periods=1).min()


thighthick_left_mean = pd.Series(thighthick_left).rolling(window=window_size, min_periods=1).mean()
thighthick_left_max = pd.Series(thighthick_left).rolling(window=window_size, min_periods=1).max()
thighthick_left_min = pd.Series(thighthick_left).rolling(window=window_size, min_periods=1).min()

shinthick_left_mean = pd.Series(shinthick_left).rolling(window=window_size, min_periods=1).mean()
shinthick_left_max = pd.Series(shinthick_left).rolling(window=window_size, min_periods=1).max()
shinthick_left_min = pd.Series(shinthick_left).rolling(window=window_size, min_periods=1).min()

footthick_left_mean = pd.Series(footthick_left).rolling(window=window_size, min_periods=1).mean()
footthick_left_max = pd.Series(footthick_left).rolling(window=window_size, min_periods=1).max()
footthick_left_min = pd.Series(footthick_left).rolling(window=window_size, min_periods=1).min()

thighthick_right_mean = pd.Series(thighthick_right).rolling(window=window_size, min_periods=1).mean()
thighthick_right_max = pd.Series(thighthick_right).rolling(window=window_size, min_periods=1).max()
thighthick_right_min = pd.Series(thighthick_right).rolling(window=window_size, min_periods=1).min()

shinthick_right_mean = pd.Series(shinthick_right).rolling(window=window_size, min_periods=1).mean()
shinthick_right_max = pd.Series(shinthick_right).rolling(window=window_size, min_periods=1).max()
shinthick_right_min = pd.Series(shinthick_right).rolling(window=window_size, min_periods=1).min()

footthick_right_mean = pd.Series(footthick_right).rolling(window=window_size, min_periods=1).mean()
footthick_right_max = pd.Series(footthick_right).rolling(window=window_size, min_periods=1).max()
footthick_right_min = pd.Series(footthick_right).rolling(window=window_size, min_periods=1).min()

thighthick_back_mean = pd.Series(thighthick_back).rolling(window=window_size, min_periods=1).mean()
thighthick_back_max = pd.Series(thighthick_back).rolling(window=window_size, min_periods=1).max()
thighthick_back_min = pd.Series(thighthick_back).rolling(window=window_size, min_periods=1).min()

shinthick_back_mean = pd.Series(shinthick_back).rolling(window=window_size, min_periods=1).mean()
shinthick_back_max = pd.Series(shinthick_back).rolling(window=window_size, min_periods=1).max()
shinthick_back_min = pd.Series(shinthick_back).rolling(window=window_size, min_periods=1).min()

footthick_back_mean = pd.Series(footthick_back).rolling(window=window_size, min_periods=1).mean()
footthick_back_max = pd.Series(footthick_back).rolling(window=window_size, min_periods=1).max()
footthick_back_min = pd.Series(footthick_back).rolling(window=window_size, min_periods=1).min()

thighthick_front_mean = pd.Series(thighthick_front).rolling(window=window_size, min_periods=1).mean()
thighthick_front_max = pd.Series(thighthick_front).rolling(window=window_size, min_periods=1).max()
thighthick_front_min = pd.Series(thighthick_front).rolling(window=window_size, min_periods=1).min()

shinthick_front_mean = pd.Series(shinthick_front).rolling(window=window_size, min_periods=1).mean()
shinthick_front_max = pd.Series(shinthick_front).rolling(window=window_size, min_periods=1).max()
shinthick_front_min = pd.Series(shinthick_front).rolling(window=window_size, min_periods=1).min()

footthick_front_mean = pd.Series(footthick_front).rolling(window=window_size, min_periods=1).mean()
footthick_front_max = pd.Series(footthick_front).rolling(window=window_size, min_periods=1).max()
footthick_front_min = pd.Series(footthick_front).rolling(window=window_size, min_periods=1).min()

# Create x-values (steps)
x = np.arange(len(body)) * 1024

# Plotting
plt.figure(figsize=(10, 6))
plt.fill_between(x, body_min, body_max, color='firebrick', alpha=0.2)
# Left
plt.fill_between(x, thighthick_left_min, thighthick_left_max, color='orange', alpha=0.2)
plt.fill_between(x, shinthick_left_min, shinthick_left_max, color='royalblue', alpha=0.2)
plt.fill_between(x, footthick_left_min, footthick_left_max, color='forestgreen', alpha=0.2)

# Right
plt.fill_between(x, thighthick_right_min, thighthick_right_max, color='red', alpha=0.2)
plt.fill_between(x, shinthick_right_min, shinthick_right_max, color='blue', alpha=0.2)
plt.fill_between(x, footthick_right_min, footthick_right_max, color='green', alpha=0.2)

# Front
plt.fill_between(x, thighthick_front_min, thighthick_front_max, color='gold', alpha=0.2)
plt.fill_between(x, shinthick_front_min, shinthick_front_max, color='deepskyblue', alpha=0.2)
plt.fill_between(x, footthick_front_min, footthick_front_max, color='limegreen', alpha=0.2)

# Back
plt.fill_between(x, thighthick_back_min, thighthick_back_max, color='darkorange', alpha=0.2)
plt.fill_between(x, shinthick_back_min, shinthick_back_max, color='navy', alpha=0.2)
plt.fill_between(x, footthick_back_min, footthick_back_max, color='darkgreen', alpha=0.2)

plt.plot(x, body_mean, 'firebrick', linewidth=2, label='Body radius')
# Left
plt.plot(x, thighthick_left_mean, 'orange', linewidth=2, label='Thigh thickness (Left)')
plt.plot(x, shinthick_left_mean, 'royalblue', linewidth=2, label='Leg thickness (Left)')
plt.plot(x, footthick_left_mean, 'forestgreen', linewidth=2, label='Foot thickness (Left)')

# Right
plt.plot(x, thighthick_right_mean, 'red', linewidth=2, label='Thigh thickness (Right)')
plt.plot(x, shinthick_right_mean, 'blue', linewidth=2, label='Leg thickness (Right)')
plt.plot(x, footthick_right_mean, 'green', linewidth=2, label='Foot thickness (Right)')

# Front
plt.plot(x, thighthick_front_mean, 'gold', linewidth=2, label='Thigh thickness (Front)')
plt.plot(x, shinthick_front_mean, 'deepskyblue', linewidth=2, label='Leg thickness (Front)')
plt.plot(x, footthick_front_mean, 'limegreen', linewidth=2, label='Foot thickness (Front)')

# Back
plt.plot(x, thighthick_back_mean, 'darkorange', linewidth=2, label='Thigh thickness (Back)')
plt.plot(x, shinthick_back_mean, 'navy', linewidth=2, label='Leg thickness (Back)')
plt.plot(x, footthick_back_mean, 'darkgreen', linewidth=2, label='Foot thickness (Back)')


# Adjust font size for tick labels
plt.tick_params(axis='both', which='major', labelsize=12)

# Add labels and legend
plt.axhline(y=0.0, color='black', linestyle='--')
plt.axhline(y=0.4, color='black', linestyle='--')
plt.xlabel('No. of Samples', fontsize = 12)
plt.ylabel('Parameter Value', fontsize  = 12)
plt.legend(ncol=2,loc='upper right', fontsize = 12)
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(10, 6))
# Left
plt.fill_between(x, thigh_left_min, thigh_left_max, color='orange', alpha=0.2)
plt.fill_between(x, shin_left_min, shin_left_max, color='royalblue', alpha=0.2)
plt.fill_between(x, foot_left_min, foot_left_max, color='forestgreen', alpha=0.2)

# Right
plt.fill_between(x, thigh_right_min, thigh_right_max, color='red', alpha=0.2)
plt.fill_between(x, shin_right_min, shin_right_max, color='blue', alpha=0.2)
plt.fill_between(x, foot_right_min, foot_right_max, color='green', alpha=0.2)

# Front
plt.fill_between(x, thigh_front_min, thigh_front_max, color='gold', alpha=0.2)
plt.fill_between(x, shin_front_min, shin_front_max, color='deepskyblue', alpha=0.2)
plt.fill_between(x, foot_front_min, foot_front_max, color='limegreen', alpha=0.2)

# Back
plt.fill_between(x, thigh_back_min, thigh_back_max, color='darkorange', alpha=0.2)
plt.fill_between(x, shin_back_min, shin_back_max, color='navy', alpha=0.2)
plt.fill_between(x, foot_back_min, foot_back_max, color='darkgreen', alpha=0.2)

# Left
plt.plot(x, thigh_left_mean, 'orange', linewidth=2, label='Thigh (Left)')
plt.plot(x, shin_left_mean, 'royalblue', linewidth=2, label='Leg (Left)')
plt.plot(x, foot_left_mean, 'forestgreen', linewidth=2, label='Foot (Left)')

# Right
plt.plot(x, thigh_right_mean, 'red', linewidth=2, label='Thigh (Right)')
plt.plot(x, shin_right_mean, 'blue', linewidth=2, label='Leg (Right)')
plt.plot(x, foot_right_mean, 'green', linewidth=2, label='Foot (Right)')

# Front
plt.plot(x, thigh_front_mean, 'gold', linewidth=2, label='Thigh (Front)')
plt.plot(x, shin_front_mean, 'deepskyblue', linewidth=2, label='Leg (Front)')
plt.plot(x, foot_front_mean, 'limegreen', linewidth=2, label='Foot (Front)')

# Back
plt.plot(x, thigh_back_mean, 'darkorange', linewidth=2, label='Thigh (Back)')
plt.plot(x, shin_back_mean, 'navy', linewidth=2, label='Leg (Back)')
plt.plot(x, foot_back_mean, 'darkgreen', linewidth=2, label='Foot (Back)')


# Add labels and legend

# Adjust font size for tick labels
plt.tick_params(axis='both', which='major', labelsize=12)

plt.axhline(y=0.4, color='black', linestyle='--')
plt.axhline(y=2.0, color='black', linestyle='--')
plt.xlabel('No. of Samples', fontsize = 12)
plt.ylabel('Parameter Value', fontsize = 12)
plt.legend(ncol=2, loc='lower right',fontsize = 12)
plt.grid(True)
plt.show()