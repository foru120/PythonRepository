import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5
n_steps = 2

X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

X_batch = np.array([[[0, 1, 2], [9, 8, 7]],
                    [[3, 4, 5], [0, 0, 0]],
                    [[6, 7, 8], [6, 5, 4]],
                    [[9, 0, 1], [3, 2, 1]]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_val = sess.run(outputs, feed_dict={X: X_batch})
    print(output_val)