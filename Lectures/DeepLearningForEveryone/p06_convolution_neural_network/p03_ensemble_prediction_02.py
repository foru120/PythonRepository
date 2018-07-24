########################################################################################################################
## ▣ 가중치, 편향 파라미터 초기화
##  - weight 는 layer 의 입출력 node 수에 따라 적응적으로 normal distribution 의 variance 를 정해주는 것이 좋다.
##  - Bias 는 아주 작은 상수값으로 초기화 해주는 것이 낫다.
##  - 따라서, weight 초기화 방법 후보로 normal, truncated_normal, xavier, he 방법을 선정하고,
##    bias 초기화 방법 후보로 normal, zero 방법을 선정하였다.
##  - no batch normalization 인 경우 he weight 에 bias 0 으로 초기화 한 경우가 가장 성능이 좋았다.
##  - batch normalization 인 경우에는 no batch normalization 인 경우보다 He 초기값인 경우 약 3~4 % 정도 성능 향상이 있다.
##  ⊙ 초기화 방법
##   1. with constant
##    - tf.Variable(tf.zeros([784, 10])) : 0 으로 초기화
##    - tf.Variable(tf.constant(0.1, [784, 10])) : 0.1 로 초기화
##   2. with normal distribution
##    - tf.Variable(tf.random_normal([784, 10])) : 평균 0, 표준편차 1 인 정규분포 값
##   3. with truncated normal distribution
##    - tf.truncated_normal([784, 10], stddev=0.1) : 평균 0, 표준편차 0.1 인 정규분포에서 샘플링 된 값이 2*stddev 보다 큰 경우 해당 샘플을 버리고 다시 샘플링하는 방법.
##   4. with Xavier initialization
##    - tf.get_variable('w1', shape=[784, 10], initializer=tf.contrib.layers.xavier_initializer())
##   5. with He initialization
##    - tf.get_variable('w1', shape=[784, 10], initializer=tf.contrib.layers.variance_scaling_initializer())
##
## ▣ tf.nn.conv2d(
##   input,                  : 4-D 입력 값 [batch, in_height, in_width, in_channels]
##   filter,                 : 4-D 필터 값 [filter_height, filter_width, in_channels, out_channels]
##   strides,                : 길이 4의 1-D 텐서. (4차원 입력이어서 각 차원마다 스트라이드 값을 설정), 기본적으로 strides = [1, stride, stride, 1] 로 설정한다.
##   padding,                : 'SAME' or 'VALID' 둘 중의 하나의 값을 가진다. (스트라이드가 1x1 인 경우에만 동작.)
##   use_cudnn_on_gpu=None,  : GPU 사용에 대한 bool 값.
##   data_format=None,       : 'NHWC' : [batch, height, width, channels], 'NCHW' : [batch, channels, height, width]
##   name=None               : 연산에 대한 이름 설정.
##   )
##  1. 2-D matrix 형태로 필터를 납작하게 만든다. (filter_height * filter_width * in_channels, output_channels]
##  2. 가상 텐서 형태로 형상화하기 위해 입력 텐서로부터 이미지 패치들을 추출한다. [batch, out_height, out_width, filter_height * filter_width * in_channels]
##  3. 각 패치에 대해 필터 행렬과 이미지 패치 벡터를 오른쪽으로 행렬곱 연산을 수행한다.
##
## ▣ tf.nn.max_pool(
##   value,             : 4-D 텐서 형태 [batch, height, width, channels], type : tf.float32
##   ksize,             : 입력 값의 각 차원에 대한 윈도우 크기.
##   strides,           : 입력 값의 각 차원에 대한 sliding 윈도우 크기.
##   padding,           : 'SAME' :  output size => input size, 'VALID' : output size => ksize - 1
##   data_format='NHWC' : 'NHWC' : [batch, height, width, channels], 'NCHW' : [batch, channels, height, width]
##   name=None          : 연산에 대한 이름 설정.
##   )
##  1. 입력 값에 대해 윈도우 크기 내에서의 가장 큰 값을 골라서 차원을 축소 시키는 함수.
##
## ▣ 경사 감소법
##  1. SGD : 이전 가중치 매개 변수에 대한 손실 함수 기울기는 수치 미분을 사용해 구하고 기울기의 학습률만큼 이동하도록 구현하는 최적화 알고리즘.
##           wi ← wi − η(∂E / ∂wi), η : 학습률
##   - tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
##  2. Momentum
##   - tf.train.MomentumOptimizer
##  3. AdaGrad
##   - tf.train.AdagradOptimizer
##  4. ADAM
##   - tf.train.AdamOptimizer
##  5. Adadelta
##   - tf.train.AdadeltaOptimizer
##  6. RMSprop
##   - tf.train.RMSPropOptimizer
##  7. Etc
##   - tf.train.AdagradDAOptimizer
##   - tf.train.FtrlOptimizer
##   - tf.train.ProximalGradientDescentOptimizer
##   - tf.train.ProximalAdagradOptimizer
########################################################################################################################
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                ########################################################################################################
                ## ▣ Dropout
                ##  - 랜덤으로 노드를 삭제하여 입력과 출력 사이의 연결을 제거하는 기법.
                ##  - 모델이 데이터에 오버피팅 되는 것을 막아주는 역할.
                ##  ⊙ 학습 : 0.5, 테스트 : 1
                ########################################################################################################
                self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')

                self.X = tf.placeholder(tf.float32, [None, 784], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 28, 28, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10], name='y_data')

            ############################################################################################################
            ## ▣ Convolution 계층 - 1
            ##  ⊙ 합성곱 계층 → filter: (3, 3), output: 32 개, 초기값: He
            ##  ⊙ 편향        → shape: 32, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 풀링 계층   → Max Pooling
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            with tf.name_scope('conv_layer1') as scope:
                self.W1 = tf.get_variable(name='W1', shape=[3, 3, 1, 32], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b1 = tf.Variable(tf.constant(value=0.001, shape=[32]), name='b1')
                self.L1 = tf.nn.conv2d(input=X_img, filter=self.W1, strides=[1, 1, 1, 1], padding='SAME')
                # self.L1 = tf.nn.relu(self.L1, name='R1')
                self.L1 = self.parametric_relu(self.L1 + self.b1, 'R1')
                self.L1 = tf.nn.max_pool(value=self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 28x28 -> 14x14
                self.L1 = tf.layers.dropout(inputs=self.L1, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## ▣ Convolution 계층 - 2
            ##  ⊙ 합성곱 계층 → filter: (3, 3), output: 64 개, 초기값: He
            ##  ⊙ 편향        → shape: 64, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 풀링 계층   → Max Pooling
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            with tf.name_scope('conv_layer2') as scope:
                self.W2 = tf.get_variable(name='W2', shape=[3, 3, 32, 64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b2 = tf.Variable(tf.constant(value=0.001, shape=[64]), name='b2')
                self.L2 = tf.nn.conv2d(input=self.L1, filter=self.W2, strides=[1, 1, 1, 1], padding='SAME')
                # self.L2 = tf.nn.relu(self.L2, name='R2')
                self.L2 = self.parametric_relu(self.L2 + self.b2, 'R2')
                self.L2 = tf.nn.max_pool(value=self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 14x14 -> 7x7
                self.L2 = tf.layers.dropout(inputs=self.L2, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## ▣ Convolution 계층 - 3
            ##  ⊙ 합성곱 계층 → filter: (3, 3), output: 128 개, 초기값: He
            ##  ⊙ 편향        → shape: 128, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 풀링 계층   → Max Pooling
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            with tf.name_scope('conv_layer3') as scope:
                self.W3 = tf.get_variable(name='W3', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b3 = tf.Variable(tf.constant(value=0.001, shape=[128]), name='b3')
                self.L3 = tf.nn.conv2d(input=self.L2, filter=self.W3, strides=[1, 1, 1, 1], padding='SAME')
                # self.L3 = tf.nn.relu(self.L3, name='R3')
                self.L3 = self.parametric_relu(self.L3 + self.b3, 'R3')
                self.L3 = tf.nn.max_pool(value=self.L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 7x7 -> 4x4
                self.L3 = tf.layers.dropout(inputs=self.L3, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## ▣ Convolution 계층 - 4
            ##  ⊙ 합성곱 계층 → filter: (3, 3), output: 256 개, 초기값: He
            ##  ⊙ 편향        → shape: 256, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 풀링 계층   → X
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            with tf.name_scope('conv_layer4') as scope:
                self.W4 = tf.get_variable(name='W4', shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b4 = tf.Variable(tf.constant(value=0.001, shape=[256]), name='b4')
                self.L4 = tf.nn.conv2d(input=self.L3, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME')
                # self.L4 = tf.nn.relu(self.L4, name='R4')
                self.L4 = self.parametric_relu(self.L4 + self.b4, 'R4')
                self.L4 = tf.layers.dropout(inputs=self.L4, rate=self.dropout_rate, training=self.training)
                self.L4 = tf.reshape(self.L4, shape=[-1, 4 * 4 * 256])

            ############################################################################################################
            ## ▣ fully connected 계층 - 1
            ##  ⊙ 가중치      → shape: (4 * 4 * 256, 625), output: 625 개, 초기값: He
            ##  ⊙ 편향        → shape: 625, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            with tf.name_scope('fc_layer1') as scope:
                self.W5 = tf.get_variable(name='W5', shape=[4 * 4 * 256, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b5 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b5'))
                # self.L5 = tf.nn.relu(tf.matmul(self.L4, self.W5) + self.b5, name='R5')
                self.L5 = self.parametric_relu(tf.matmul(self.L4, self.W5) + self.b5, 'R5')
                self.L5 = tf.layers.dropout(inputs=self.L5, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## ▣ fully connected 계층 - 2
            ##  ⊙ 가중치      → shape: (625, 625), output: 625 개, 초기값: He
            ##  ⊙ 편향        → shape: 625, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            with tf.name_scope('fc_layer2') as scope:
                self.W6 = tf.get_variable(name='W6', shape=[625, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b6 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b6'))
                # self.L6 = tf.nn.relu(tf.matmul(self.L5, self.W6) + self.b6, name='R6')
                self.L6 = self.parametric_relu(tf.matmul(self.L5, self.W6) + self.b6, 'R6')
                self.L6 = tf.layers.dropout(inputs=self.L6, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## ▣ 출력층
            ##  ⊙ 가중치      → shape: (625, 10), output: 10 개, 초기값: He
            ##  ⊙ 편향        → shape: 10, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Softmax
            ############################################################################################################
            self.W7 = tf.get_variable(name='W7', shape=[625, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b7 = tf.Variable(tf.constant(value=0.001, shape=[10], name='b7'))
            self.logits = tf.matmul(self.L6, self.W7) + self.b7

        ################################################################################################################
        ## ▣ L2-Regularization
        ##  ⊙ λ/(2*N)*Σ(W)²-> (0.001/(2*tf.to_float(tf.shape(self.X)[0])))*tf.reduce_sum(tf.square(self.W7))
        ################################################################################################################
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + (0.001/(2*tf.to_float(tf.shape(self.X)[0])))*tf.reduce_sum(tf.square(self.W7))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

        # self.tensorflow_summary()

    ####################################################################################################################
    ## ▣ Tensorboard logging
    ##  ⊙ tf.summary.histogram : 여러개의 행렬 값을 logging 하는 경우
    ##  ⊙ tf.summary.scalar : 한개의 상수 값을 logging 하는 경우
    ####################################################################################################################
    def tensorflow_summary(self):
        self.W1_hist = tf.summary.histogram('W1_conv1', self.W1)
        self.b1_hist = tf.summary.histogram('b1_conv1', self.b1)
        self.L1_hist = tf.summary.histogram('L1_conv1', self.L1)

        self.W2_hist = tf.summary.histogram('W2_conv2', self.W2)
        self.b2_hist = tf.summary.histogram('b2_conv2', self.b2)
        self.L2_hist = tf.summary.histogram('L2_conv2', self.L2)

        self.W3_hist = tf.summary.histogram('W3_conv3', self.W3)
        self.b3_hist = tf.summary.histogram('b3_conv3', self.b3)
        self.L3_hist = tf.summary.histogram('L3_conv3', self.L3)

        self.W4_hist = tf.summary.histogram('W4_conv4', self.W4)
        self.b4_hist = tf.summary.histogram('b4_conv4', self.b4)
        self.L4_hist = tf.summary.histogram('L4_conv4', self.L4)

        self.W5_hist = tf.summary.histogram('W5_fc1', self.W5)
        self.b5_hist = tf.summary.histogram('b5_fc1', self.b5)
        self.L5_hist = tf.summary.histogram('L5_fc1', self.L5)

        self.W6_hist = tf.summary.histogram('W6_fc2', self.W6)
        self.b6_hist = tf.summary.histogram('b6_fc2', self.b6)
        self.L6_hist = tf.summary.histogram('L6_fc2', self.L6)

        self.cost_hist = tf.summary.scalar(self.name+'/cost_hist', self.cost)
        self.accuracy_hist = tf.summary.scalar(self.name+'/accuracy_hist', self.accuracy)

        # ※ merge_all 로 하는 경우, hist 를 모으지 않는 변수들도 대상이 되어서 에러가 발생한다.
        #    따라서 merge 로 모으고자하는 변수를 각각 지정해줘야한다.
        self.merged = tf.summary.merge([self.W1_hist, self.b1_hist, self.L1_hist,
                                        self.W2_hist, self.b2_hist, self.L2_hist,
                                        self.W3_hist, self.b3_hist, self.L3_hist,
                                        self.W4_hist, self.b4_hist, self.L4_hist,
                                        self.W5_hist, self.b5_hist, self.L5_hist,
                                        self.W6_hist, self.b6_hist, self.L6_hist,
                                        self.cost_hist, self.accuracy_hist])

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    ####################################################################################################################
    ## ▣ Parametric Relu or Leaky Relu
    ##  ⊙ alpha 값을 설정해 0 이하인 경우 alpha 만큼의 경사를 설정해서 0 이 아닌 값을 리턴하는 함수
    ##  ⊙ Parametric Relu : 학습을 통해 최적화된 alpha 값을 구해서 적용하는 Relu 함수
    ##  ⊙ Leaky Relu      : 0 에 근접한 작은 값을 alpha 값으로 설정하는 Relu 함수
    ####################################################################################################################
    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

training_epochs = 20
batch_size = 100

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

sess = tf.Session()

models = []
num_models = 5
for m in range(num_models):
    models.append(Model(sess, 'model' + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

import time
# 시작 시간 체크
stime = time.time()

for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)

    # train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 각각의 모델 훈련
        for idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[idx] += c / total_batch
            # train_writer.add_summary(s)
    print('Epoch: ', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
print('Learning Finished!')

# 테스트 모델에서 정확도(accuracy) 체크
# test_x, test_y = mnist.test.next_batch(1000)
# test_size = len(test_y)
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)

model_result = np.zeros(test_size*2, dtype=np.int).reshape(test_size, 2)
model_result[:, 0] = range(0, test_size)

for idx, m in enumerate(models):
    # print(idx, 'Accuracy: ', m.get_accuracy(test_x, test_y))
    # p = m.predict(test_x)
    print(idx, 'Accuracy: ', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    model_result[:, 1] = np.argmax(p, 1)
    for result in model_result:
        predictions[result[0], result[1]] += 1

# ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y, 1))
ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble Accuracy: ', sess.run(ensemble_accuracy))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))