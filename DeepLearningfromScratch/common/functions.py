import numpy as np

# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

# ###################################################################################################
# ## 퍼셉트론(단층 & 다층)
# ###################################################################################################
# # AND Gate
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#
# # NAND Gate
# def NAND(x1, x2):
#     x = np.array([x1, x2])
#     w = np.array([-0.5, -0.5])
#     b = 0.7
#     return 1 if np.sum(x*w)+b > 0 else 0
#
# # OR Gate
# def OR(x1, x2):
#     x = np.array([x1, x2])
#     w = np.array([0.5, 0.5])
#     b = -0.4
#     return 1 if np.sum(x*w)+b > 0 else 0
#
# # XOR Gate
# def XOR(x1, x2):
#     return AND(OR(x1, x2), NAND(x1, x2))
#
#
# ###################################################################################################
# ## 활성화 함수
# ###################################################################################################
# # 계단 함수
# def step_function(x):
#     return np.array(x > 0, dtype=np.int)
#
# # 시그모이드 함수(출력층에서 2-Class 인 경우)
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
# # 소프트맥스 함수(출력층에서 Multi-Class 인 경우)
# def softmax(a):
#     C = np.max(a)
#     exp_a = np.exp(a-C)
#     sum_exp_a = np.sum(exp_a)
#     return exp_a/sum_exp_a
#
# # 항등 함수(회귀 분석인 경우)
# def relu_function(x):
#     return np.maximum(0, x)
#
#
# ###################################################################################################
# ## 미분 관련 함수
# ###################################################################################################
# # 수치 미분
# def numerical_diff(f, x):
#     h = 1e-4  # 0.0001
#     return (f(x+h)-f(x-h))/(2*h)
#
# 편 미분
# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성
#
#     it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
#
#     while not it.finished:
#         idx = it.multi_index
#
#         # f(x+h) 계산
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1 = f(x)
#
#         # f(x-h) 계산
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
#
#         # f(x+h) - f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2*h)
#
#         x[idx] = tmp_val
#         it.iternext()
#
#     return grad
#
# # 경사 하강법
# def gradient_descent(f, init_x, lr=0.01, step_num=10000):
#     x = init_x
#
#     for i in range(step_num):
#         grad = numerical_gradient(f, x)
#         print(grad, x)
#         x -= lr * grad
#
#
# ###################################################################################################
# ## 손실 함수
# ###################################################################################################
# # 평균제곱오차 함수
# def mean_squared_error(y, t):
#     y = np.array(y)
#     t = np.array(t)
#     return np.sum(np.power(y-t, 2))/2
#
# # 교차 엔트로피 함수(배치용)
# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     delta = 1e-7
#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y+delta)) / batch_size