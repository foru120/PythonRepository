# 문제1.텐서플로우를 이용하지 않고 파이썬으로 단층 신경망을 구현하시오.
# coding: utf-8
import sys, os
import numpy as np

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from DeepLearningClass.common.layers import *
from DeepLearningClass.common.gradient import numerical_gradient
from collections import OrderedDict
import matplotlib.pyplot as plt
from DeepLearningClass.dataset.mnist import load_mnist

class TwoLayerNet:
    #    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    def __init__(self, input_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)
        # self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # self.params['b2'] = np.zeros(output_size)
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        #  self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        #  grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network = TwoLayerNet(input_size=784, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]  # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch)  # 600

for i in range(iters_num):  # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size)  # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신

    for key in ('W1', 'b1'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)  # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0:  # 600 번마다 정확도 쌓는다.
        print(x_train.shape)  # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)  # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 문제2.단층 신경망을 텐서플로우로 구현하시오.(p .112)
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
# mnist데이터의 실제 y라벨을 넣기 위한 빈 공간

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 비용함수를 구현하는데 여기서 사용되는 reduce_sum 은 차원축소 후 sum하는 함수

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# learning rate=0.01  SGD경사하강법으로 비용함수의 오차를 최소화시킨다.

sess = tf.Session()
# 텐서플로우 그래프 연산을 시작하게끔 세션객체를 생성한다.
sess.run(tf.global_variables_initializer())
# 모든 변수를 초기화한다.

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 훈련 데이터 셋에서 무작위로 100개를 추출
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # 위에서 생성한 100개의 데이터를 SGD로 훈련시킨다
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # y라벨 중 가장 큰 인덱스를 리턴하고 y_(실제값) 중 가장 큰 인덱스를 리턴해서 같은지 비교한다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# tf.cast [True, False, True...]   =>   [1,0,1,....] 로 변경해서 reduce_mean구함
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# 문제2.위의 단층 신경망 코드의 중요문법 3가지
# 1. reduce_sum과 reduce_mean 예제:
x = np.arange(6).reshape(2, 3)
print(x)

sess = tf.Session()
# print(sess.run(tf.reduce_sum(x))) = > 1차원으로 축소해서 합쳤다. 결과: 15

# 예제: 열 단위로 sum하기
x = np.arange(6).reshape(2, 3)
print(x)

sess = tf.Session()
print(sess.run(tf.reduce_sum(x, 0)))

# 문제3.(점심시간 문제) 위의 결과를 행단위로 sum하게 하시오.
x = np.arange(6).reshape(2, 3)
print(x)

sess = tf.Session()
print(sess.run(tf.reduce_sum(x, 1)))

# 문제4.숫자 0 으로 채워진 2 행 3 열의 행렬을 만들고 숫자 1 로 채워진 2 행 3 열의 행렬을 만들고, 두 행렬의 합을 출력하시오.
sess = tf.Session()
a = tf.zeros([2, 3])
b = tf.ones([2, 3])
print(sess.run(tf.add(a, b)))
# sess=tf.Session()  앞까지는 아무것도 실행되지 않는다. 세션을 생성하여 run() 메소드를 호출해야 비로소 심볼릭 코드가 실제 실행된다.

# p .42, 43 수학 함수 / 행렬 연산 함수 참고

# 문제5.숫자 2 로 채워진 2 행 3 열의 행렬을 만들고 숫자 3 으로 채워진 2 행 3 열의 행렬을 만들고 두 행렬의 행렬합을 구하시오.
sess = tf.Session()
a = 2 * tf.ones([2, 3])
b = 3 * tf.ones([2, 3])
print(sess.run(tf.add(a, b)))

import tensorflow as tf

# a = tf.zeros([2, 3])
# b = tf.ones([2, 3])
a = tf.Variable([[2,2,2], [2,2,2]])
b = tf.Variable([[3,3,3], [3,3,3]])
c = a + b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))

print('====================================================================================================')
print('== 문제 6. 숫자 2로 채워진 2x3 행렬과 숫자 3으로 채워진 3x2 행렬의 행렬곱을 출력하시오!')
print('====================================================================================================\n')
a = tf.Variable([[2,2,2], [2,2,2]])
b = tf.Variable([[3,3,], [3,3,], [3,3]])
c = tf.matmul(a,b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))

print('====================================================================================================')
print('== 문제 8. 위에서 출력한 100개의 숫자의 평균값을 출력하시오!')
print('====================================================================================================\n')
import tensorflow as tf

correct_prediction = [ True, False , True  ,True  ,True  ,True  ,True,  True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True, False , True  ,True, False , True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True,
  True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True ,False , True  ,True  ,True  ,True  ,True
  ,True  ,True, False , True, False , True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
 ,False , True  ,True  ,True]

a = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))

