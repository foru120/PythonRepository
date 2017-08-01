# epoch - 0 , train_acc - 0.162633333333 , test_acc - 0.1643
# epoch - 1 , train_acc - 0.908066666667 , test_acc - 0.9118
# epoch - 2 , train_acc - 0.92915 , test_acc - 0.9309
# epoch - 3 , train_acc - 0.944583333333 , test_acc - 0.9433
# epoch - 4 , train_acc - 0.952916666667 , test_acc - 0.9502
# epoch - 5 , train_acc - 0.958583333333 , test_acc - 0.9559
# epoch - 6 , train_acc - 0.963966666667 , test_acc - 0.9598
# epoch - 7 , train_acc - 0.967416666667 , test_acc - 0.9636
# epoch - 8 , train_acc - 0.971666666667 , test_acc - 0.9665
# epoch - 9 , train_acc - 0.9745 , test_acc - 0.969
# epoch - 10 , train_acc - 0.976133333333 , test_acc - 0.9695
# epoch - 11 , train_acc - 0.978616666667 , test_acc - 0.9711
# epoch - 12 , train_acc - 0.980516666667 , test_acc - 0.9725
# epoch - 13 , train_acc - 0.9817 , test_acc - 0.9728
# epoch - 14 , train_acc - 0.982633333333 , test_acc - 0.9739
# epoch - 15 , train_acc - 0.984766666667 , test_acc - 0.9753
# epoch - 16 , train_acc - 0.984633333333 , test_acc - 0.974
import sys, os

sys.path.append(os.pardir)

import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.chapter5.two_layer_net_5_layer import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size1=100, hidden_size2=100, hidden_size3=100, hidden_size4=100, output_size=10)

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