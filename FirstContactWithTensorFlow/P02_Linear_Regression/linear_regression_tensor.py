import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))  # 평균 제곱 오차

optimizer = tf.train.GradientDescentOptimizer(0.5)  # 경사 감소법
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # 텐서 변수 초기화

sess = tf.Session()  # 텐서 세션 생성
sess.run(init)

for step in range(20):
    sess.run(train)
    print(step, sess.run(W), sess.run(b), sess.run(loss))

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.xlabel('x')
plt.ylabel('y')
plt.show()