# coding: utf-8
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T

def cross_entropy_error(y, t):
    t = t.argmax(axis=1)  # 정답 label 이 one-hot encoding 이므로 최대값만 구한다
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
