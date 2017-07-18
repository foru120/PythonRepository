import numpy as np

def mean_squared_error(y, t):
    y = np.array(y)
    t = np.array(t)
    return np.sum(np.power(y-t, 2))/2

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

print(mean_squared_error(y, t))