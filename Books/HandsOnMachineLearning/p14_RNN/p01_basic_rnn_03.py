import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5
n_steps = 2

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
outputs = tf.transpose(tf.stack(outputs_seqs), perm=[1, 0, 2])

X_batch = np.array([[[0, 1, 2], [9, 8, 7]],
                    [[3, 4, 5], [0, 0, 0]],
                    [[6, 7, 8], [6, 5, 4]],
                    [[9, 0, 1], [3, 2, 1]]])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})
    print(outputs_val)