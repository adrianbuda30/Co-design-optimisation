import numpy as np


accumulated_rewards_chopping_metric = [[4   , 4  ,  4  , 4 ,   4 , 4], 
                                       [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5], 
                                       [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5], 
                                       [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5], 
                                       [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]]


mean_rewards = [np.mean(accumulated_rewards_chopping_metric[i]) for i in range(len(accumulated_rewards_chopping_metric))]

print(mean_rewards)