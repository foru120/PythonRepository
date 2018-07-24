import numpy as np
import matplotlib.pylab as plt

# 수치 미분
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h)-f(x-h))/(2*h)
    # 나쁜 구현의 예
    # h = 10e-50
    # return (f(x+h)-f(x))/h

def function_1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y2, linestyle='--')
tf = tangent_line(function_1, 10)
y3 = tf(x)
plt.plot(x, y3, linestyle='--')

plt.show()