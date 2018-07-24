# epoch - 0 , train_acc - 0.293316666667 , test_acc - 0.2883
# epoch - 1 , train_acc - 0.894283333333 , test_acc - 0.9027
# epoch - 2 , train_acc - 0.905616666667 , test_acc - 0.9116
# epoch - 3 , train_acc - 0.90975 , test_acc - 0.9142
# epoch - 4 , train_acc - 0.914316666667 , test_acc - 0.9168
# epoch - 5 , train_acc - 0.916266666667 , test_acc - 0.9182
# epoch - 6 , train_acc - 0.91685 , test_acc - 0.9202
# epoch - 7 , train_acc - 0.918983333333 , test_acc - 0.9204
# epoch - 8 , train_acc - 0.919 , test_acc - 0.921
# epoch - 9 , train_acc - 0.920633333333 , test_acc - 0.9207
# epoch - 10 , train_acc - 0.921533333333 , test_acc - 0.9221
# epoch - 11 , train_acc - 0.922016666667 , test_acc - 0.9215
# epoch - 12 , train_acc - 0.922233333333 , test_acc - 0.9223
# epoch - 13 , train_acc - 0.923383333333 , test_acc - 0.9235
# epoch - 14 , train_acc - 0.924483333333 , test_acc - 0.9215
# epoch - 15 , train_acc - 0.9241 , test_acc - 0.9222
# epoch - 16 , train_acc - 0.925566666667 , test_acc - 0.9225
import sys, os

sys.path.append(os.pardir)

import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.chapter5.two_layer_net_1_layer import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784,output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    for key in network.params.keys():
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('epoch -', int(i / iter_per_epoch), ', train_acc -', train_acc, ', test_acc -', test_acc)