import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

from LearningTensorFlow.Chapter4_Convolutional_Neural_Network.Mnist_Neural_Network.layers import *

class CifarLoader(object):
    DATA_PATH = '/home/kyh/dataset/cifar-10-batches-py'

    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None  # (50,000, 32, 32, 3)
        self.labels = None  # (10,000, 10)

    def _unpickle(self, file):
        with open(os.path.join(CifarLoader.DATA_PATH, file), 'rb') as fo:
            u = pickle._Unpickler(fo)
            u.encoding = 'latin1'
            dict = u.load()
        return dict

    def _one_hot(self, vec, vals=10):
        out = np.zeros((vec.shape[0], vals))
        out[range(vec.shape[0]), vec] = 1
        return out

    def load(self):
        data = [self._unpickle(f) for f in self._source]  # dict_keys(['batch_label', 'labels', 'data', 'filenames'])
        images = np.vstack([d['data'] for d in data])
        self.images = images.reshape(images.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = self._one_hot(np.hstack([d['labels'] for d in data]), 10)

    def next_batch(self, batch_size):
        x, y = self.images[self._i: self._i + batch_size], self.labels[self._i: self._i + batch_size]
        self._i = (self._i + batch_size) % self.images.shape[0]
        return x, y

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(['data_batch_{}'.format(i) for i in range(1, 6)])
        self.train.load()
        self.test = CifarLoader(['test_batch'])
        self.test.load()
        print('>> CIFAR10 Dataset Loaded!!!')

    def display_cifar(self, images, size):
        plt.figure()
        plt.gca().set_axis_off()
        im = np.vstack([np.hstack([images[np.random.choice(images.shape[0])] for _ in range(size)]) for _ in range(size)])
        plt.imshow(im)
        plt.show()

class Model(object):
    def __init__(self, sess):
        self.sess = sess
        self._build_graph()

    def _build_graph(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None)

        conv1_1 = conv_layer(input=self.x, shape=[3, 3, 3, 30])
        conv1_2 = conv_layer(input=conv1_1, shape=[3, 3, 30, 30])
        conv1_3 = conv_layer(input=conv1_2, shape=[3, 3, 30, 30])
        conv1_pool = max_pool_2x2(conv1_3)
        conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=self.keep_prob)

        conv2_1 = conv_layer(input=conv1_drop, shape=[3, 3, 30, 50])
        conv2_2 = conv_layer(input=conv2_1, shape=[3, 3, 50, 50])
        conv2_3 = conv_layer(input=conv2_2, shape=[3, 3, 50, 50])
        conv2_pool = max_pool_2x2(conv2_3)
        conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=self.keep_prob)

        conv3_1 = conv_layer(input=conv2_drop, shape=[3, 3, 50, 80])
        conv3_2 = conv_layer(input=conv3_1, shape=[3, 3, 80, 80])
        conv3_3 = conv_layer(input=conv3_2, shape=[3, 3, 80, 80])
        conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

        conv3_flat = tf.reshape(conv3_pool, shape=[-1, 1*1*80])
        conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=self.keep_prob)

        full_1 = tf.nn.relu(full_layer(input=conv3_drop, size=500))
        full1_drop = tf.nn.dropout(full_1, keep_prob=self.keep_prob)

        logit = full_layer(input=full1_drop, size=10)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.y_))
        self.opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit, 1), tf.argmax(self.y_, 1)), tf.float32))

    def train(self, train_x, train_y):
        return self.sess.run([self.acc, self.loss, self.opt], feed_dict={self.x: train_x, self.y_: train_y, self.keep_prob: 0.6})

    def test(self, test_x, test_y):
        return self.sess.run(self.acc, feed_dict={self.x: test_x, self.y_: test_y, self.keep_prob: 1.0})

#todo CIFAR10 데이터 셋 확인
# d = CifarDataManager()
# print('Number of train images: {}'.format(d.train.images.shape[0]))
# print('Number of train labels: {}'.format(d.train.labels.shape[0]))
# print('Number of test images: {}'.format(d.test.images.shape[0]))
# print('Number of test labels: {}'.format(d.test.labels.shape[0]))
# d.display_cifar(d.train.images, 10)

#todo CIFAR10 신경망 수행
STEPS = 5000
BATCH_SIZE = 100

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
)

with tf.Session(config=config) as sess:
    cifar = CifarDataManager()
    model = Model(sess=sess)

    sess.run(tf.global_variables_initializer())

    # Train Part
    for i in range(STEPS):
        batch_x, batch_y = cifar.train.next_batch(BATCH_SIZE)
        train_acc, train_loss, _ = model.train(batch_x, batch_y)

        if i % 100 == 0:
            print('>> Train step: {}, acc: {}, loss: {}'.format(i, train_acc, train_loss, 5))

    # Test Part
    test_x = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    test_y = cifar.test.labels.reshape(10, 1000, 10)

    test_acc = np.mean(np.asarray([model.test(x, y) for x, y in zip(test_x, test_y)]))
    print('>> Test Acc: {}'.format(test_acc, 5))