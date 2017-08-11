from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])

fc1 = tf.contrib.layers.fully_connected(x, 100)
output = tf.contrib.layers.fully_connected(x, 10)

y = tf.nn.softmax(output)
y_ = tf.placeholder("float", [None, 10])
# mnist데이터의 실제 y라벨을 넣기 위한 빈 공간

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 비용함수를 구현하는데 여기서 사용되는 reduce_sum 은 차원축소 후 sum하는 함수

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# learning rate=0.01  SGD경사하강법으로 비용함수의 오차를 최소화시킨다.

sess = tf.Session()
# 텐서플로우 그래프 연산을 시작하게끔 세션객체를 생성한다.
sess.run(tf.global_variables_initializer())
# 모든 변수를 초기화한다.

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 훈련 데이터 셋에서 무작위로 100개를 추출
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # 위에서 생성한 100개의 데이터를 SGD로 훈련시킨다
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # y라벨 중 가장 큰 인덱스를 리턴하고 y_(실제값) 중 가장 큰 인덱스를 리턴해서 같은지 비교한다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# tf.cast [True, False, True...]   =>   [1,0,1,....] 로 변경해서 reduce_mean구함
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))