import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import numpy as np

best_obj = float("inf")
best_params = None

def obj(params: pd.DataFrame) -> np.ndarray:
    x1 = params['x1'].values
    x2 = params['x2'].values
    x3 = params['x3'].values
    x4 = params['x4'].values
    reward = np.linalg.norm((np.array([x1, x2, x3, x4]).T - np.array([10, 20, 30, 40])), axis=1)
    return reward.reshape(-1, 1)

space = DesignSpace().parse([
    {'name': 'x1', 'type': 'num', 'lb': 10, 'ub': 100},
    {'name': 'x2', 'type': 'num', 'lb': 10, 'ub': 100},
    {'name': 'x3', 'type': 'num', 'lb': 10, 'ub': 100},
    {'name': 'x4', 'type': 'num', 'lb': 10, 'ub': 100}
])

opt = HEBO(space)

for i in range(100):
    rec = opt.suggest(n_suggestions = 8)
    current_obj = obj(rec)
    print(current_obj)
    print(rec)
    opt.observe(rec, current_obj)
    
    min_current_obj = current_obj.min()
    
    if min_current_obj < best_obj:
        best_obj = min_current_obj
        best_params = rec.loc[current_obj.argmin()].values
    
    # print(f'After {i+1} iterations, best obj is {best_obj:.2f}')
    # print('Best params so far:', best_params)
    # print('Current params:', rec.values[7])
