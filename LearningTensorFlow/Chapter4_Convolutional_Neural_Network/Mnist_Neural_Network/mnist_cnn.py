from LearningTensorFlow.Chapter4_Convolutional_Neural_Network.Mnist_Neural_Network.layers import *

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '/home/kyh/PycharmProjects/PythonRepository/LearningTensorFlow/Chapter4_Convolutional_Neural_Network/Mnist_Neural_Network/data'
STEPS = 5000

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = conv_layer(input=x_image, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(x=conv1)

conv2 = conv_layer(input=conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        tr_batch_xs, tr_batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: tr_batch_xs, y_: tr_batch_ys, keep_prob: 0.6})

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: tr_batch_xs, y_: tr_batch_ys, keep_prob: 1.0})
            print('>> step {}, training accuracy {}'.format(i, train_accuracy))

            tot_test_accuracy = []
            for _ in range(mnist.test.num_examples // 50):
                ts_batch_xs, ts_batch_ys = mnist.test.next_batch(50)
                test_accuracy = sess.run(accuracy, feed_dict={x: ts_batch_xs, y_: ts_batch_ys, keep_prob: 1.0})
                tot_test_accuracy.append(test_accuracy)

            print('>> step {}, test accuracy {}'.format(i, np.mean(np.asarray(tot_test_accuracy))))