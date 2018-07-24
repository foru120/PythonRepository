# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from DeepLearningClass.common.layers import *
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 표준 정규 분포를 따르는 난수 생성
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()  # forward, backward 시 계층 순서대로 수행하기 위해 순서가 있는 OrderedDict 를 사용.
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():  # Affine1 -> Relu1 -> Affine2
            x = layer.forward(x)  # 각 계층마다 forward 수행
        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):  # x : (100, 1024), t : (100, 10)
        y = self.predict(x)  # (100, 10) : 마지막 출력층을 통과한 신경망이 예측한 값
        return self.lastLayer.forward(y, t)  # 마지막 계층인 SoftmaxWithLoss 계층에 대해 forward 수행

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # [[0.1, 0.05, 0.5, 0.05, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1], ....] -> [2, 4, 2, 1, 9, ....]
        if t.ndim != 1: t = np.argmax(t, axis=1)  # t.ndim != 1 이면 one-hot encoding 인 경우이므로, 2차원 배열로 값이 들어온다

        accuracy = np.mean(y == t)
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()  # 역전파를 수행하기 위해 기존 layer 순서를 반대로 바꾼다.
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads