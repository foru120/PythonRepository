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
                self.dropout_rate = tf.Variable(tf.constant(value=0.7), name='dropout_rate')
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
            ##  ⊙ 가중치      → shape: (7 * 7 * 256, 625), output: 625 개, 초기값: He
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

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + 0.01*tf.reduce_sum(tf.square(self.W7))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

        self.tensorflow_summary()

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
        return self.sess.run([self.merged, self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

training_epochs = 20
batch_size = 100

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

sess = tf.Session()

models = []
num_models = 2
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

    train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 각각의 모델 훈련
        for idx, m in enumerate(models):
            s, c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[idx] += c / total_batch
            train_writer.add_summary(s)
    print('Epoch: ', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
print('Learning Finished!')

# 테스트 모델에서 정확도(accuracy) 체크
test_x, test_y = mnist.test.next_batch(1000)
test_size = len(test_y)
# test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)

for idx, m in enumerate(models):
    print(idx, 'Accuracy: ', m.get_accuracy(test_x, test_y))
    p = m.predict(test_x)
    # print(idx, 'Accuracy: ', m.get_accuracy(mnist.test.images, mnist.test.labels))
    # p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y, 1))
# ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble Accuracy: ', sess.run(ensemble_accuracy))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))