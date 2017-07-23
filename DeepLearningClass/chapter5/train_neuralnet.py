# coding: utf-8
import sys, os

sys.path.append(os.pardir)

import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.chapter5.two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]  # 60000 개
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)  # train_size 가 batch_size 보다 작으면 1보다 작은 수가 나오므로, 최소 1 epoch 을 돌기 위해 1 과 비교

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)  # 0 ~ train_size(60,000) 사이의 값 중에 batch_size 만큼 랜덤으로 선택
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad, loss = network.gradient(x_batch, t_batch)  # 오차역전파법 방식

    # Weight, Bias 갱신
    for key in network.params.keys():
        network.params[key] -= learning_rate * grad[key]

    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:  # 매 epoch 마다 수행
        train_acc = network.accuracy(x_train, t_train)  # 전체 train 데이터에 대해 정확도를 구함
        test_acc = network.accuracy(x_test, t_test)  # 전체 test 데이터에 대해 정확도를 구함
        train_acc_list.append(train_acc)  # 매 epoch 마다 구한 train 데이터의 정확도를 저장
        test_acc_list.append(test_acc)  # 매 epoch 마다 구한 test 데이터의 정확도를 저장
        print(train_acc, test_acc)