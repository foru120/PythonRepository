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
    return (f(x+delta)-f(x-delta))/(2+delta)

def func(x):
    return 0.001 * x**2 + 0.1 * x

print(numerical_diff(func, 10))