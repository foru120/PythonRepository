from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# read_data_sets() : mnist.train : 훈련 데이터, mnist.test : 테스트 데이터가 들어 있는 데이터 셋을 가져온다.
#                    데이터의 각 원소는 이미지(xs)와 레이블(ys)로 구성되어 있다.
#                    훈련 이미지는 mnist.train.images 로 참조가 가능하고, 훈련 레이블은 mnist.train.labels 로 참조가 가능하다.
#                    각각의 이미지의 픽셀들은 0~1 사이의 값을 가진다. (0으로 갈수록 흰색, 1로 갈수록 검은색)
#                    mnist.train.images -> (55000, 784)
#                    mnist.train.labels -> (55000, 10)
mnist = input_data.read_data_sets('MNIST_data\\', one_hot=True)

# 1. 입력층
#  - 입력값에 가중치 W 를 곱하고 편향 b 를 더해 그 다음 계층으로 넘겨주는 계층.
#  - tf.zeros(shape) : shape 차원의 상수 텐서를 생성하면서 초기 값은 0 으로 지정.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder('float', [None, 784])  # shape 크기가 None 은 어떤 크기나 가능하다는 뜻.

# 2. 출력층
#  - affine 계층으로 연산된 값이 들어오고, 해당 값을 가지고 활성화 함수를 적용해 출력하는 계층.
#  - tf.matmul(x, W) : 두 텐서를 행렬곱셈하여 결과 텐서를 리턴. (affine 계층)
#  - tf.nn.softmax() : 활성화 함수인 softmax 함수를 구현하는 함수.
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder('float', [None, 10])

# 3. 학습을 위한 손실함수 구현
#  - 평균 제곱 오차 : loss = tf.reduce_mean(tf.square(y-y_data))
#  - 교차 엔트로피  : cross_entropy = -tf.reduce_sum(tf.multiply(y_, tf.log(y)))
#  - tf.multiply(a, b) : 두 개의 텐서간의 행렬곱을 구해주는 함수.
#  - tf.reduce_sum(tensor) : axis 를 설정하지 않으면 전체 sum 결과를 출력하는 함수.
cross_entropy = -tf.reduce_sum(tf.multiply(y_, tf.log(y)))

# 4. 경사 감소법
#  - 여기서는 가장 일반적인 경사감소법인 SGD(확률적 경사 하강법)를 사용했다.
#  - tf.train.GradientDescentOptimizer(HyperParameter) : HyperParameter 값에 대해 경사 감소를 수행.
#  - optimizer.minimize(error_rate) : 최소가 되는 error_rate 를 찾아주는 함수.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 5. 학습 수행
#  - mnist.train.next_batch(batch_cnt) : MNIST 데이터 셋에서 batch_cnt 만큼 데이터를 추출.
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 6. 정확도 검증
#  - tf.equal(tensor1, tensor2) : tensor1 과 tensor2 의 요소들을 비교해서 같으면 True, 다르면 False 를 리턴하는 함수.
#  - tf.cast(tensor, type) : tensor 를 특정 type 으로 변형하는 함수
#  - tf.reduce_mean(tensor) : axis 축 값이 없으면 전체 데이터에 대한 평균을 구한다.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))  # 마지막 테스트 데이터에 대한 정확도를 검증한다.

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))