thighthick_front_mean = pd.Series(thighthick_front).rolling(window=window_size, min_periods=1).mean()
thighthick_front_max = pd.Series(thighthick_front).rolling(window=window_size, min_periods=1).max()
thighthick_front_min = pd.Series(thighthick_front).rolling(window=window_size, min_periods=1).min()

shinthick_front_mean = pd.Series(shinthick_front).rolling(window=window_size, min_periods=1).mean()
shinthick_front_max = pd.Series(shinthick_front).rolling(window=window_size, min_periods=1).max()
shinthick_front_min = pd.Series(shinthick_front).rolling(window=window_size, min_periods=1).min()

footthick_front_mean = pd.Series(footthick_front).rolling(window=window_size, min_periods=1).mean()
footthick_front_max = pd.Series(footthick_front).rolling(window=window_size, min_periods=1).max()
footthick_front_min = pd.Series(footthick_front).rolling(window=window_size, min_periods=1).min()