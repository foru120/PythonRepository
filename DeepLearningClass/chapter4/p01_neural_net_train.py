print('====================================================================================================')
print('== 문제 66. 위의 one-hot encoding 된 t 값과 확률 y, y2 를 각각 평균제곱오차 함수(비용함수)에 입력해서 오차가 '
      '어떤게 더 낮은지 출력하시오!')
print('====================================================================================================\n')
import numpy as np
def mean_squared_error(y, t):
    return 0.5*np.mean(np.square(y - t), dtype=np.float32)

t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(mean_squared_error(y, t))

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))
t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(cross_entropy_error(y, t))

print('====================================================================================================')
print('== 문제 67. (점심시간 문제) 아래의 넘파이 배열을 교차 엔트로피 오차 함수를 이용해서 오차율이 어떻게 되는지'
      'for loop 문을 사용해서 한번에 알아내게 하시오.')
print('====================================================================================================\n')
t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
y = np.array([[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.1,0.0,0.0],
              [0.1,0.05,0.2,0.0,0.05,0.1,0.0,0.6,0.0,0.0],
              [0.0,0.05,0.3,0.0,0.05,0.1,0.0,0.6,0.0,0.0],
              [0.0,0.05,0.4,0.0,0.05,0.0,0.0,0.5,0.0,0.0],
              [0.0,0.05,0.5,0.0,0.05,0.0,0.0,0.4,0.0,0.0],
              [0.0,0.05,0.6,0.0,0.05,0.0,0.0,0.3,0.0,0.0],
              [0.0,0.05,0.7,0.0,0.05,0.0,0.0,0.2,0.0,0.0],
              [0.0,0.1,0.8,0.0,0.1,0.0,0.0,0.2,0.0,0.0],
              [0.0,0.05,0.9,0.0,0.05,0.0,0.0,0.0,0.0,0.0]])

for y_ in y:
    print(cross_entropy_error(y_, t))

print('====================================================================================================')
print('== 문제 68. 60000 미만의 숫자중에서 무작위로 10개를 출력하시오.')
print('====================================================================================================\n')
a = np.random.choice(range(0, 60000), 10)
print(a)

import numpy as np
import pickle
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.common.functions import sigmoid, softmax

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
y_batch = t_train[batch_mask]

print(len(x_batch))
print(x_batch.shape)

print('====================================================================================================')
print('== 문제 70. 데이터 1개를 가지고 오차를 구하는 교차 엔트로피 값을 구하시오.')
print('====================================================================================================\n')
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta)) / len(y)

print(cross_entropy_error(y, t))

print('====================================================================================================')
print('== 문제 71. 데이터 10개를 가지고 오차를 구하는 교차 엔트로피 값을 구하시오.')
print('====================================================================================================\n')
import sys, os
sys.path.append(os.pardir)
import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = 60000
batch_size = 10

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def  softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta)) / len(y)

def init_network():
    with open("DeepLearningClass/chapter4/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

cost_list = []

for _ in range(100000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = t_train[batch_mask]

    y_ = predict(init_network(), x_batch)
    c = cross_entropy_error(y_, y_batch)
    cost_list.append(c)

print(np.max(cost_list))

print('====================================================================================================')
print('== 문제 72. 근사로 구한 미분 함수를 파이썬으로 구현하시오!')
print('====================================================================================================\n')
def numerical_diff(f, x):
    delta = 1e-4
    return (f(x+delta)-f(x-delta))/(2*delta)

def func(x):
    return 0.001 * x**2 + 0.1 * x

print(numerical_diff(func, 10))

print('====================================================================================================')
print('== NCS 문제 1. 아래의 함수를 x = 7 에서 수치미분하면 기울기가 어떻게 되는가?')
print('====================================================================================================\n')
def numerical_diff(f, x):
    delta = 1e-4
    return (f(x+delta)-f(x-delta))/(2*delta)

def func(x):
    return 3 * x**2 + 4 * x

print(numerical_diff(func, 4))

print('====================================================================================================')
print('== NCS 문제 2. 아래의 행렬 x 의 100 개의 각각의 행에서 가장 큰 원소를 빼서 다시 x2 라는 변수에 저장하시오!')
print('====================================================================================================\n')
import numpy as np
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

x = np.random.rand(100, 784)
x2 = np.argmax(x, axis=1)
print(x2)

# ■ 4.5 편미분
#
# 변수가 2개 이상인 함수를 미분할 때 미분 대상 변수 외 나머지 변수를 상수처럼 고정시켜 미분하는 것을 편미분이라고 한다.
# f(x0,x1) = x0^2 + x1^2 의 그래프를 보면 아래와 같다.
print('====================================================================================================')
print('== 문제 76. f(x0,x1) = x0^2 + x1^2 이 함수를 편미분하는데 x0가 3이고 x1이 4일 때 구하시오.')
print('====================================================================================================\n')
print(numerical_diff(samplefunction3,3))

print('====================================================================================================')
print('== 문제 77. x0가 3이고 x1이 4일 때 아래의 함수를 아래와 같이 편미분하시오.')
print('====================================================================================================\n')
print(numerical_diff(samplefunction3,4))

print('====================================================================================================')
print('== 문제 78. 아래의 함수를 x0로 편미분하시오.')
print('====================================================================================================\n')
def numerical_diff(f,x):
    h=0.0001
    return (f(x+h)-f(x-h))/(2*h)

def samplefunction3(x):
    return 2*x**2

print(numerical_diff(samplefunction3,3))

print('====================================================================================================')
print('== 문제 79. 아래의 함수를 x1에 대해 편미분하시오. (x0=6, x1=7)  : lambda식 이용하기')
print('====================================================================================================\n')
def numerical_diff(f,x):
    h=0.0001
    return (f(x+h)-f(x-h))/(2*h)

func = lambda x0:2*x0**2
print(numerical_diff(func,6))

print('====================================================================================================')
print('== 문제 80. (점심시간 문제) for loop 문을 이용해서 아래의 함수를 x0로 편미분하고 x1로 편미분이 각각 수행되게 하시오.')
print('====================================================================================================\n')
def f1(x):
    return 6*x**2
def f2(x):
    return 2*x**2

def numerical_diff(f, x):
    h = 0.0001
    return (f(x+h)-f(x-h))/(2*h)

funcs = [f1, f2]
xs = [6.0, 7.0]
for i in range(2):
    print(numerical_diff(funcs[i], xs[i]))

print('====================================================================================================')
print('== 문제 81. 위의 편미분을 코딩하시오.')
print('====================================================================================================\n')
import numpy as np

def numerical_gradient(f, x):
    h = 0.0001
    grad = np.zeros_like(x)  # x 와 형상이 같은 배열 grad 가 만들어짐

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def samplefunc4(x):
    return x[0]**2 + x[1]**2

print(numerical_gradient(samplefunc4, np.array([3.0, 4.0])))

print('====================================================================================================')
print('== 문제 82. np.zeros_like 가 무엇인지 확인해보시오')
print('====================================================================================================\n')
def numerical_gradient(f, x):
    h = 0.0001
    grad = np.zeros_like(x)  # x 와 형상이 같은 배열 grad 가 만들어짐
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def samplefunc4(x):
    return x[0] ** 2 + x[1] ** 2

print('====================================================================================================')
print('== 문제 83. x0=3.0, x1=0.0 일때의 기울기 벡터를 구하시오.')
print('====================================================================================================\n')
print(numerical_gradient(samplefunc4, np.array([3.0, 0.0])))

print('====================================================================================================')
print('== 문제 85. 경사 감소 함수를 파이썬으로 구현하시오.')
print('====================================================================================================\n')
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grade = numerical_gradient(f, x)
        x -= lr * grade
    return x

init_x = np.array([-3.0, 4.0])

def function_2(x):
    return x[0]**2 + x[1]**2
print(gradient_descent(function_2, init_x, lr=10))
print(gradient_descent(function_2, init_x, lr=1e-10))

print('====================================================================================================')
print('== 문제 86. 위의 식을 그대로 사용해서 테스트를 수행하는데 학습률이 너무 크면 발산을하고 학습률이 너무 작으면 수렴을 못'
      '한다는 것을 테스트하시오.')
print('====================================================================================================\n')
def numerical_gradient(f, x):
    h = 0.0001
    grad = np.zeros_like(x)  # x 와 형상이 같은 배열 grad 가 만들어짐
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grade = numerical_gradient(f, x)
        x -= lr * grade
    return x

init_x = np.array([-3.0, 4.0])

def function_2(x):
    return x[0]**2 + x[1]**2
print(gradient_descent(function_2, init_x, lr=10))

print('====================================================================================================')
print('== 문제 87. learning rate 를 1e-10 으로 했을 때 기울기가 0으로 수렴하려면 step_num 을 몇으로 줘야하는지 확인히시오.')
print('====================================================================================================\n')