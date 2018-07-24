import sys, os
sys.path.append(os.pardir)
import numpy as np
from DeepLearningfromScratch.common.functions import softmax, cross_entropy_error, numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)

net = simpleNet()
print(net.W)  # 가중치 매개변수
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p, np.argmax(p))
t = np.array([0, 0, 1])  # 정답 레이블
print(net.loss(x, t))

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)