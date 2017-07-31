# epoch - 0 , train_acc - 0.195766666667 , test_acc - 0.1965
# epoch - 1 , train_acc - 0.922583333333 , test_acc - 0.9198
# epoch - 2 , train_acc - 0.9356 , test_acc - 0.9341
# epoch - 3 , train_acc - 0.943983333333 , test_acc - 0.939
# epoch - 4 , train_acc - 0.946316666667 , test_acc - 0.9451
# epoch - 5 , train_acc - 0.949716666667 , test_acc - 0.9456
# epoch - 6 , train_acc - 0.951533333333 , test_acc - 0.9478
# epoch - 7 , train_acc - 0.954033333333 , test_acc - 0.948
# epoch - 8 , train_acc - 0.957016666667 , test_acc - 0.9518
# epoch - 9 , train_acc - 0.956933333333 , test_acc - 0.9523
# epoch - 10 , train_acc - 0.959516666667 , test_acc - 0.955
# epoch - 11 , train_acc - 0.958983333333 , test_acc - 0.953
# epoch - 12 , train_acc - 0.9611 , test_acc - 0.9535
# epoch - 13 , train_acc - 0.961033333333 , test_acc - 0.9549
# epoch - 14 , train_acc - 0.9624 , test_acc - 0.9539
# epoch - 15 , train_acc - 0.96285 , test_acc - 0.9541
# epoch - 16 , train_acc - 0.96315 , test_acc - 0.9541
import sys, os

sys.path.append(os.pardir)

import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.chapter5.two_layer_net_batch_norm import TwoLayerNet

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