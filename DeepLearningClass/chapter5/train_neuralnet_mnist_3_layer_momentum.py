# epoch - 0 , train_acc - 0.0754 , test_acc - 0.0728
# epoch - 1 , train_acc - 0.86505 , test_acc - 0.865
# epoch - 2 , train_acc - 0.9139 , test_acc - 0.9139
# epoch - 3 , train_acc - 0.938466666667 , test_acc - 0.9385
# epoch - 4 , train_acc - 0.95845 , test_acc - 0.9538
# epoch - 5 , train_acc - 0.967166666667 , test_acc - 0.9631
# epoch - 6 , train_acc - 0.971666666667 , test_acc - 0.9654
# epoch - 7 , train_acc - 0.97515 , test_acc - 0.9669
# epoch - 8 , train_acc - 0.978633333333 , test_acc - 0.9683
# epoch - 9 , train_acc - 0.982266666667 , test_acc - 0.9711
# epoch - 10 , train_acc - 0.984766666667 , test_acc - 0.9729
# epoch - 11 , train_acc - 0.985766666667 , test_acc - 0.9733
# epoch - 12 , train_acc - 0.986483333333 , test_acc - 0.9726
# epoch - 13 , train_acc - 0.989583333333 , test_acc - 0.9761
# epoch - 14 , train_acc - 0.991133333333 , test_acc - 0.9736
# epoch - 15 , train_acc - 0.990016666667 , test_acc - 0.9744
# epoch - 16 , train_acc - 0.993816666667 , test_acc - 0.9761
import sys, os

sys.path.append(os.pardir)

import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.chapter5.two_layer_net_3_layer import TwoLayerNet
from DeepLearningClass.common.optimizer import Momentum

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

momentum = Momentum()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    momentum.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('epoch -', int(i / iter_per_epoch), ', train_acc -', train_acc, ', test_acc -', test_acc)