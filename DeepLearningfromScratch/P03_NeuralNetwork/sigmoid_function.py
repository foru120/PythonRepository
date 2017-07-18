import numpy as np
import matplotlib.pylab as plb

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

x1 = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x1)

plb.plot(x, y, label='sigmoid')
plb.plot(x1, y1, linestyle='--', label='step_func')
plb.ylim(-0.1, 1.1)
plb.show()