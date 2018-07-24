# epoch - 0 , train_acc - 0.56145 , test_acc - 0.5525
# epoch - 1 , train_acc - 0.974416666667 , test_acc - 0.9664
# epoch - 2 , train_acc - 0.983666666667 , test_acc - 0.97
# epoch - 3 , train_acc - 0.988983333333 , test_acc - 0.9777
# epoch - 4 , train_acc - 0.990483333333 , test_acc - 0.9769
# epoch - 5 , train_acc - 0.992716666667 , test_acc - 0.9803
# epoch - 6 , train_acc - 0.995566666667 , test_acc - 0.9815
# epoch - 7 , train_acc - 0.995916666667 , test_acc - 0.9788
# epoch - 8 , train_acc - 0.997683333333 , test_acc - 0.983
# epoch - 9 , train_acc - 0.997466666667 , test_acc - 0.981
# epoch - 10 , train_acc - 0.9982 , test_acc - 0.9823
# epoch - 11 , train_acc - 0.9982 , test_acc - 0.982
# epoch - 12 , train_acc - 0.998683333333 , test_acc - 0.9821
# epoch - 13 , train_acc - 0.998016666667 , test_acc - 0.9808
# epoch - 14 , train_acc - 0.999116666667 , test_acc - 0.9825
# epoch - 15 , train_acc - 0.999433333333 , test_acc - 0.9831
# epoch - 16 , train_acc - 0.9997 , test_acc - 0.9842
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
    network.is_train = True

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
        network.is_train = False
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('epoch -', int(i / iter_per_epoch), ', train_acc -', train_acc, ', test_acc -', test_acc)