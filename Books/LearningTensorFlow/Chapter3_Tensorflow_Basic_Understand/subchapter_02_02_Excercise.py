import tensorflow as tf

""" A
"""
a = tf.constant(value=5, dtype=tf.int32)
b = tf.constant(value=10, dtype=tf.int32)
c = tf.multiply(a, b)
d = tf.add(a, b)
e = tf.subtract(c, d)
f = tf.add(c, d)
g = tf.divide(e, f)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(g))

""" B
"""
a = tf.constant(value=5., dtype=tf.float32)
b = tf.constant(value=10., dtype=tf.float32)
c = tf.multiply(a, b)
d = tf.sin(c)
e = tf.divide(d, b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(e))