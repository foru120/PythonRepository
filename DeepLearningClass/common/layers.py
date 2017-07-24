# coding: utf-8
import numpy as np
from DeepLearningClass.common.functions import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # 0 을 기준으로 True, False 값을 추출
        out = x.copy()
        out[self.mask] = 0  # 0 보다 밑인 값들을 0으로 할당

        return out

    def backward(self, dout):
        dout[self.mask] = 0  # 역전파 시에 x < 0 이하는 미분값이 0 이므로, x < 0 인 부분은 0으로 할당해준다
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None

        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        # 순전파 ▷ X : (2, 3), W : (3, 4) -> XㆍY : (2, 4)
        # 역전파 ▷ XㆍY : (2, 4) -> X : (2, 4)ㆍWＴ, W : XＴㆍ(2, 4)
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)  # 편향은 순전파시 각각의 데이터에 더해지므로, 역전파시에 각 축의 값이 편향의 원소에 모여야 한다

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx