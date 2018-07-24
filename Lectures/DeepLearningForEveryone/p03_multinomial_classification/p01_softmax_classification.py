########################################################################################################################
## ▣ softmax classification
##  - 여러개의 값으로 분류할 때 사용하는 기법
##  - softmax(여러 출력값을 확률로 변경시키는 함수), 교차 엔트로피
########################################################################################################################
import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder('float', [None, 4])
Y = tf.placeholder('float', [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predict = tf.cast(tf.equal(tf.arg_max(hypothesis, dimension=1), tf.arg_max(Y, dimension=1)), tf.float32)
accuracy = tf.reduce_mean(predict)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data}))