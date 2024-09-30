import numpy as np
import torch.distributions as D

max_arm_length = [10]
min_arm_length = [0.5]

for _ in range(100000):
    print(D.Normal(0.5, 0.5).sample())
    



# initial_mean = np.array([[2.31], [2.77], [1.87], [3.73], [2.32], [2.32], [1.51], [3.35]])
# initial_std = np.array([[0.58], [0.62], [0.26], [0.46], [0.54], [0.58], [0.47], [0.57]])
# print(initial_mean[0])
