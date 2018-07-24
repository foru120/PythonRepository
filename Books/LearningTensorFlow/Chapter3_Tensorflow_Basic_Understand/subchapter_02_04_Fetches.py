import tensorflow as tf

a = tf.constant(value=5, dtype=tf.int32)
b = tf.constant(value=10, dtype=tf.int32)
c = tf.multiply(a, b)
d = tf.add(a, b)
e = tf.subtract(c, d)
f = tf.add(c, d)

with tf.Session() as sess:
    fetches = [a, b, c, d, e, f]
    outs = sess.run(fetches)

print("outs = {}".format(outs))
print(type(outs[0]))