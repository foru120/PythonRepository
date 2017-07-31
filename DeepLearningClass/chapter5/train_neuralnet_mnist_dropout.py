# epoch - 0 , train_acc - 0.106433333333 , test_acc - 0.1068
# epoch - 1 , train_acc - 0.112366666667 , test_acc - 0.1135
# epoch - 2 , train_acc - 0.565033333333 , test_acc - 0.5685
# epoch - 3 , train_acc - 0.791066666667 , test_acc - 0.7991
# epoch - 4 , train_acc - 0.8489 , test_acc - 0.8519
# epoch - 5 , train_acc - 0.878283333333 , test_acc - 0.8841
# epoch - 6 , train_acc - 0.895616666667 , test_acc - 0.892
# epoch - 7 , train_acc - 0.907616666667 , test_acc - 0.905
# epoch - 8 , train_acc - 0.92065 , test_acc - 0.9187
# epoch - 9 , train_acc - 0.92455 , test_acc - 0.9242
# epoch - 10 , train_acc - 0.9316 , test_acc - 0.9271
# epoch - 11 , train_acc - 0.9357 , test_acc - 0.9329
# epoch - 12 , train_acc - 0.938666666667 , test_acc - 0.9375
# epoch - 13 , train_acc - 0.94205 , test_acc - 0.9362
# epoch - 14 , train_acc - 0.9457 , test_acc - 0.9449
# epoch - 15 , train_acc - 0.948633333333 , test_acc - 0.9443
# epoch - 16 , train_acc - 0.950066666667 , test_acc - 0.9462
import sys, os

sys.path.append(os.pardir)

import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.chapter5.two_layer_net_dropout import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size1=200, hidden_size2=200, output_size=10)

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