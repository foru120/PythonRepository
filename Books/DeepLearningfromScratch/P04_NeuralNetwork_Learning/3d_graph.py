import matplotlib.pylab as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def function_2(x):
    return x[0]**2 + x[1]**2

x = np.arange(-4.0, 4.0, 0.1)
y = np.arange(-4.0, 4.0, 0.1)
X, Y = np.meshgrid(x, y)
Z = function_2((X, Y))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z,
                       rstride=2,
                       cstride=2,
                       cmap=cm.RdPu,
                       linewidth=1,
                       antialiased=True)

ax.set_title('3D Graph')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=30, azim=70)
ax.dist = 8
plt.show()