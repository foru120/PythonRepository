# epoch - 0 , train_acc - 0.2669 , test_acc - 0.2775
# epoch - 1 , train_acc - 0.945066666667 , test_acc - 0.9449
# epoch - 2 , train_acc - 0.966366666667 , test_acc - 0.9615
# epoch - 3 , train_acc - 0.977516666667 , test_acc - 0.9698
# epoch - 4 , train_acc - 0.982216666667 , test_acc - 0.9718
# epoch - 5 , train_acc - 0.983283333333 , test_acc - 0.9723
# epoch - 6 , train_acc - 0.989316666667 , test_acc - 0.9778
# epoch - 7 , train_acc - 0.987283333333 , test_acc - 0.9751
# epoch - 8 , train_acc - 0.99195 , test_acc - 0.9775
# epoch - 9 , train_acc - 0.994066666667 , test_acc - 0.9779
# epoch - 10 , train_acc - 0.992933333333 , test_acc - 0.9769
# epoch - 11 , train_acc - 0.9953 , test_acc - 0.9797
# epoch - 12 , train_acc - 0.9945 , test_acc - 0.9772
# epoch - 13 , train_acc - 0.995633333333 , test_acc - 0.9787
# epoch - 14 , train_acc - 0.996316666667 , test_acc - 0.9772
# epoch - 15 , train_acc - 0.9979 , test_acc - 0.9803
# epoch - 16 , train_acc - 0.996116666667 , test_acc - 0.9771
import sys, os

sys.path.append(os.pardir)

import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.chapter5.two_layer_net_3_layer import TwoLayerNet
from DeepLearningClass.common.optimizer import Adam

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

adam = Adam()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    adam.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('epoch -', int(i / iter_per_epoch), ', train_acc -', train_acc, ', test_acc -', test_acc)