from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import threading

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.learning_rate = 0.01
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.device('/gpu:0'):
                self.keep_prob = tf.placeholder(tf.float32)

                self.X = tf.placeholder(tf.float32, [None, 784])
                X_img = tf.reshape(self.X, shape=[-1, 28, 28, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10])

                # Convolution 계층 - 1
                self.W1 =tf.get_variable(name='W1', shape=[3, 3, 1, 32], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b1 = tf.Variable(tf.constant(value=0.001, shape=[32]), name='b1')
                self.L1 = tf.nn.conv2d(input=X_img, filter=self.W1, strides=[1, 1, 1, 1], padding='SAME')
                # self.L1 = tf.nn.relu(self.L1, name='R1')
                self.L1 = self.parametric_relu(self.L1 + self.b1)
                self.L1 = tf.nn.max_pool(value=self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 28x28 -> 14x14

                # Convolution 계층 - 2
                self.W2 = tf.get_variable(name='W2', shape=[3, 3, 32, 64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b2 = tf.Variable(tf.constant(value=0.001, shape=[64]), name='b2')
                self.L2 = tf.nn.conv2d(input=self.L1, filter=self.W2, strides=[1, 1, 1, 1], padding='SAME')
                # self.L2 = tf.nn.relu(self.L2, name='R2')
                self.L2 = self.parametric_relu(self.L2 + self.b2)
                self.L2 = tf.nn.max_pool(value=self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 14x14 -> 7x7

                # Convolution 계층 - 3
                self.W3 = tf.get_variable(name='W3', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b3 = tf.Variable(tf.constant(value=0.001, shape=[128]), name='b3')
                self.L3 = tf.nn.conv2d(input=self.L2, filter=self.W3, strides=[1, 1, 1, 1], padding='SAME')
                # self.L3 = tf.nn.relu(self.L3, name='R3')
                self.L3 = self.parametric_relu(self.L3 + self.b3)
                # self.L3 = tf.nn.max_pool(value=self.L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 7x7 -> 4x4

                # Convolution 계층 - 4
                self.W4 = tf.get_variable(name='W4', shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b4 = tf.Variable(tf.constant(value=0.001, shape=[256]), name='b4')
                self.L4 = tf.nn.conv2d(input=self.L3, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME')
                # self.L4 = tf.nn.relu(self.L4, name='R4')
                self.L4 = self.parametric_relu(self.L4 + self.b4)
                self.L4 = tf.reshape(self.L4, shape=[-1, 7 * 7 * 256])

                # fully connected 계층 - 1
                self.W5 = tf.get_variable(name='W5', shape=[7 * 7 * 256, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b5 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b5'))
                # self.L5 = tf.nn.relu(tf.matmul(self.L4, self.W5) + self.b5, name='R5')
                self.L5 = self.parametric_relu(tf.matmul(self.L4, self.W5) + self.b5)
                self.L5 = tf.nn.dropout(self.L5, keep_prob=self.keep_prob)

                # fully connected 계층 - 2
                self.W6 = tf.get_variable(name='W6', shape=[625, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b6 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b6'))
                # self.L6 = tf.nn.relu(tf.matmul(self.L5, self.W6) + self.b6, name='R6')
                self.L6 = self.parametric_relu(tf.matmul(self.L5, self.W6) + self.b6)
                self.L6 = tf.nn.dropout(self.L6, keep_prob=self.keep_prob)

                # 출력층
                self.W7 = tf.get_variable(name='W7', shape=[625, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b7 = tf.Variable(tf.constant(value=0.001, shape=[10], name='b7'))
                self.logits = tf.matmul(self.L6, self.W7) + self.b7

                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    def predict(self, x_test, keep_prob=1.0):
        with tf.device('/gpu:0'):
            return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        with tf.device('/gpu:0'):
            return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob})

    def train(self, x_data, y_data, keep_prob=0.7):
        with tf.device('/gpu:0'):
            return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob})

    def parametric_relu(self, _x):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.1),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

class ThreadHandler:
    def __init__(self, models, batch_xs=None, batch_ys=None):
        self.models = models
        self.batch_xs = batch_xs
        self.batch_ys = batch_ys
        self.avg_cost_list = np.zeros(len(self.models))

    def init_cost(self):
        self.avg_cost_list = np.zeros(len(self.models))

    def train_threading(self, m, total_batch):
        c, _ = m.train(self.batch_xs, self.batch_ys)
        self.avg_cost_list[self.models.index(m)] += c / total_batch

training_epochs = 1
batch_size = 100

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

sess = tf.Session()

models = []
num_models = 7
for m in range(num_models):
    models.append(Model(sess, 'model' + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

import time
# 시작 시간 체크
stime = time.time()

coord = tf.train.Coordinator()
handler = ThreadHandler(models)

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epochs):
    for i in range(total_batch):
        handler.batch_xs, handler.batch_ys = mnist.train.next_batch(batch_size)
        thread_list = []
        for idx, m in enumerate(models):
            thread_list.append(threading.Thread(target=handler.train_threading, args=(m, total_batch)))
        # 각각의 모델 훈련
        for t in thread_list:
            t.start()
        coord.join(thread_list)
    print('Epoch: ', '%04d' % (epoch + 1), 'cost =', handler.avg_cost_list)
    handler.init_cost()
print('Learning Finished!')

coord.request_stop()

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