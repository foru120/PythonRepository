import numpy as np
import tensorflow as tf

xy = np.loadtxt('data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 3], name='x_data')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='y_data')

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, H, train], feed_dict={X: x_data, Y: y_data})

    if step % 100 == 0:
        print(step, 'Cost : ', cost_val, '\nPredict : ', h_val)