import numpy as np
import matplotlib.pylab as plb

def relu_function(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu_function(x)

plb.plot(x, y)
plb.ylim(-0.1, 5.1)
plb.show()