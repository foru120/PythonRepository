import tensorflow as tf
import numpy as np

x_data = np.random.randn(5, 10)
w_data = np.random.randn(10, 1)

with tf.Graph().as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(5, 10))
    w = tf.placeholder(dtype=tf.float32, shape=(10, 1))
    b = tf.fill(dims=(5, 1), value=-1.)
    xw = tf.matmul(x, w)

    xwb = xw + b
    s = tf.reduce_max(xwb)

    with tf.Session() as sess:
        outs = sess.run(s, feed_dict={x: x_data, w: w_data})

print('outs = {}'.format(outs))