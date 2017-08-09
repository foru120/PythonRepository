import numpy as np

a = np.zeros(50)
b = np.ones(5)
a[:5] = 0.5 * (a[:5] + b)
print(a)

a = []
a += [1,2]
a += [3,4]
print(a)