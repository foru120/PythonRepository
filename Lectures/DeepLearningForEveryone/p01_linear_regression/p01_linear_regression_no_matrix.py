########################################################################################################################
## ■ Linear Regression
##  - 여러개의 입력 값들에 대해 x1, x2, x3 .... 추출되는 특정 y 값을 예측할 때 사용하는 기법.
##  - H(X) = WX + b
########################################################################################################################

import tensorflow as tf
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# 손실 함수 : 평균오차제곱
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 최적화 : 경사 감소법
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 텐서 변수 초기화

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})

    if step % 10 == 0:
        print(step, 'Cost : ', cost_val, '\nPrediction : ', hy_val)
