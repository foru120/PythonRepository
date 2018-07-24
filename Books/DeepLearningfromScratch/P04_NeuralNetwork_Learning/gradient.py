import numpy as np

# 편미분
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_val
    return grad

# 경사 하강법
def gradient_descent(f, init_x, lr=0.01, step_num=10000):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        print(grad, x)
        x -= lr * grad

    return x

def function_2(x):
    return x[0]**2 + x[1]**2

print(gradient_descent(function_2, np.array([-3.0, 4.0]), lr=0.1, step_num=100))