import numpy as np

a = np.zeros(50)
b = np.ones(5)
a[:5] = 0.5 * (a[:5] + b)
print(a)

np.mea