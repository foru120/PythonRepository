import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(np.dot(np.array(t), np.log(np.array(y)+delta)))

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

print(np.array(y).shape, np.array(t).shape)
print(cross_entropy_error(y, t))