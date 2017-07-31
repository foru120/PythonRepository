# epoch - 0 , train_acc - 0.134183333333 , test_acc - 0.1341
# epoch - 1 , train_acc - 0.953983333333 , test_acc - 0.9519
# epoch - 2 , train_acc - 0.96605 , test_acc - 0.9618
# epoch - 3 , train_acc - 0.97145 , test_acc - 0.965
# epoch - 4 , train_acc - 0.97645 , test_acc - 0.9697
# epoch - 5 , train_acc - 0.979716666667 , test_acc - 0.9725
# epoch - 6 , train_acc - 0.981283333333 , test_acc - 0.9729
# epoch - 7 , train_acc - 0.983833333333 , test_acc - 0.975
# epoch - 8 , train_acc - 0.9853 , test_acc - 0.9756
# epoch - 9 , train_acc - 0.985833333333 , test_acc - 0.9754
# epoch - 10 , train_acc - 0.988 , test_acc - 0.9765
# epoch - 11 , train_acc - 0.98875 , test_acc - 0.9774
# epoch - 12 , train_acc - 0.9897 , test_acc - 0.9777
# epoch - 13 , train_acc - 0.990766666667 , test_acc - 0.9775
# epoch - 14 , train_acc - 0.9916 , test_acc - 0.9774
# epoch - 15 , train_acc - 0.991683333333 , test_acc - 0.9771
# epoch - 16 , train_acc - 0.9926 , test_acc - 0.9793
import sys, os

sys.path.append(os.pardir)

import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.chapter5.two_layer_net_3_layer import TwoLayerNet
from DeepLearningClass.common.optimizer import AdaGrad

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

adagrad = AdaGrad()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    adagrad.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('epoch -', int(i / iter_per_epoch), ', train_acc -', train_acc, ', test_acc -', test_acc)