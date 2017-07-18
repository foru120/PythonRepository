########################################################################################################################
## ▣ Convolution Neural Network
##  - 기존 신경망에서 한 차원 더 높은 텐서를 가지고 신경망을 구축하는 기법.
##  - 테두리, 선, 색 등 이미지의 시각적 특징이나 특성을 감지하는 용도.
##  - 입력 데이터가 첫 번째 은닉 계층의 뉴런에 완전 연결되어 있지 않음.
##  - 입력층 -> 은닉층(스트라이드 -> 활성화 함수(Relu) -> 패딩) -> 은닉층(Affine -> 활성화 함수(Relu)) -> 출력층(활성화 함수(Softmax))
########################################################################################################################
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

conv_layer_output = [64, 128]  # 합성곱 계층 output 개수
last_layer_output = 1024  # 마지막 계층 output 개수
filter_cnt = [(5, 5), (5, 5)]  # 합성곱 계층 filter 개수
output_cnt = 10  # 출력 개수

# input_data.read_data_sets(데이터셋을 저장할 경로, one_hot=True) : Mnist Sample 데이터를 로드한다.
# one_hot encoding : 한 요소만 1 인 벡터.
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, output_cnt])

x_image = tf.reshape(x, [-1, 28, 28, 1])  # 28 : 이미지 너미, 28 : 이미지 높이, 1 : 컬러 채널

# ♣ 파라미터 초기화
#  - weight 는 layer 의 입출력 node 수에 따라 적응적으로 normal distribution 의 variance 를 정해주는 것이 좋다.
#  - Bias 는 아주 작은 상수값으로 초기화 해주는 것이 낫다.
#  - 따라서, weight 초기화 방법 후보로 normal, truncated_normal, xavier, he 방법을 선정하고,
#    bias 초기화 방법 후보로 normal, zero 방법을 선정하였다.
#  - no batch normalization 인 경우 he weight 에 bias 0 으로 초기화 한 경우가 가장 성능이 좋았다.
#  - batch normalization 인 경우에는 no batch normalization 인 경우보다 He 초기값인 경우 약 3~4 % 정도 성능 향상이 있다.

#  1. with constant
#   - tf.Variable(tf.zeros([784, 10])) : 0 으로 초기화
#   - tf.Variable(tf.constant(0.1, [784, 10])) : 0.1 로 초기화
#  2. with normal distribution
#   - tf.Variable(tf.random_normal([784, 10])) : 평균 0, 표준편차 1 인 정규분포 값
#  3. with truncated normal distribution
#   - tf.truncated_normal([784, 10], stddev=0.1) : 평균 0, 표준편차 0.1 인 정규분포에서 샘플링 된 값이 2*stddev 보다 큰 경우 해당 샘플을 버리고 다시 샘플링하는 방법.
#  4. with Xavier initialization
#   - tf.get_variable('w1', shape=[784, 10], initializer=tf.contrib.layers.xavier_initializer())
#  5. with He initialization
#   - tf.get_variable('w1', shape=[784, 10], initializer=tf.contrib.layers.variance_scaling_initializer())

# 필터 변수 초기화
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 편향 변수 초기화
def bias_variable(shape, type, name=None):
    if type == 'constant':
        initial = tf.Variable(tf.constant(0.001, shape=shape))
    elif type == 'ND':  # normal distribution
        initial = tf.Variable(tf.random_normal(shape))
    elif type == 'TND':  # truncated normal distribution
        initial = tf.truncated_normal(shape)
    elif type == 'Xavier':
        initial = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    elif type == 'He':
        initial = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    return initial

# tf.nn.conv2d(
#   input,                  : 4-D 입력 값 [batch, in_height, in_width, in_channels]
#   filter,                 : 4-D 필터 값 [filter_height, filter_width, in_channels, out_channels]
#   strides,                : 길이 4의 1-D 텐서. (4차원 입력이어서 각 차원마다 스트라이드 값을 설정), 기본적으로 strides = [1, stride, stride, 1] 로 설정한다.
#   padding,                : 'SAME' or 'VALID' 둘 중의 하나의 값을 가진다.
#   use_cudnn_on_gpu=None,  : GPU 사용에 대한 bool 값.
#   data_format=None,       : 'NHWC' : [batch, height, width, channels], 'NCHW' : [batch, channels, height, width]
#   name=None               : 연산에 대한 이름 설정.
#)
#  1. 2-D matrix 형태로 필터를 납작하게 만든다. (filter_height * filter_width * in_channels, output_channels]
#  2. 가상 텐서 형태로 형상화하기 위해 입력 텐서로부터 이미지 패치들을 추출한다. [batch, out_height, out_width, filter_height * filter_width * in_channels]
#  3. 각 패치에 대해 필터 행렬과 이미지 패치 벡터를 오른쪽으로 행렬곱 연산을 수행한다.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# tf.nn.max_pool(
#   value,             : 4-D 텐서 형태 [batch, height, width, channels], type : tf.float32
#   ksize,             : 입력 값의 각 차원에 대한 윈도우 크기.
#   strides,           : 입력 값의 각 차원에 대한 sliding 윈도우 크기.
#   padding,           : 'SAME' :  output size => input size, 'VALID' : output size => ksize - 1
#   data_format='NHWC' : 'NHWC' : [batch, height, width, channels], 'NCHW' : [batch, channels, height, width]
#   name=None          : 연산에 대한 이름 설정.
#)
#  1. 입력 값에 대해 윈도우 크기 내에서의 가장 큰 값을 골라서 차원을 축소 시키는 함수.
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ▣ 첫 번째 은닉층
#  1. 합성곱 계층
#   - 윈도우 크기가 5 x 5 인 32 개 필터 생성.
#   - 32 개 가중치 행렬에 대한 편향 정의.
W_conv1 = weight_variable([*filter_cnt[0], 1, conv_layer_output[0]])
b_conv1 = bias_variable([conv_layer_output[0]], 'constant')

#  2. 활성화 함수(ReLU)
#   - tf.nn.relu(tensor) : 음수면 0, 그 외에는 x 를 리턴하는 활성화 함수.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#  3. max pooling
#   - 현재 설정은 2x2 윈도우 크기의 max pooling 이 구현되어 있는 함수.
h_pool1 = max_pool_2x2(h_conv1)

# ▣ 두 번째 은닉층
#  1. 합성곱 계층
#   - 윈도우 크기가 5 x 5 인 64 개의 필터 생성.
#   - 64 개 가중치 행렬에 대한 편향 정의.
#   - 이전 계층의 출력 값의 크기를 채널의 수로 넘겨야 함.
W_conv2 = weight_variable([*filter_cnt[1], conv_layer_output[0], conv_layer_output[1]])
b_conv2 = bias_variable([conv_layer_output[1]], 'constant')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# ▣ 세 번째 은닉층
#  - NN 신경망에서 사용하는 방식으로 처리하는 계층
#  - 전체 이미지 처리를 위해 1024 개의 뉴런을 사용. (임의로 지정)
#  - 가중치 처음 차원의 수는 두 번째 은닉층의 출력 개수.
#  - 마지막 은닉층에서는 softmax 에서 처리하기 위해 출력 값을 2-D 로 변환해야 함.
B, H, W, C = h_pool2.get_shape().as_list()
W_fc1 = weight_variable([W*H*conv_layer_output[1], last_layer_output])
b_fc1 = bias_variable([last_layer_output], 'constant')

h_pool2_flat = tf.reshape(h_pool2, [-1, W*H*conv_layer_output[1]])  # softmax 함수는 이미지를 직렬화해서 벡터 형태로 입력해야 하므로 변환 수행.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# ▣ 드롭아웃
#  - 랜덤으로 노드를 삭제하여 입력과 출력 사이의 연결을 제거하는 기법.
#  - 모델이 데이터에 오버피팅 되는 것을 막아주는 역할.
keep_prob = tf.placeholder('float')  # 뉴런이 드롭아웃되지 않을 확률을 저장할 변수
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ▣ 출력층
#  - 이미지 분류를 위한 softmax 함수를 생성.
W_fc2 = weight_variable([last_layer_output, output_cnt])
b_fc2 = bias_variable([output_cnt], 'constant')
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# ♣ 경사 감소법
#  1. SGD : 이전 가중치 매개 변수에 대한 손실 함수 기울기는 수치 미분을 사용해 구하고 기울기의 학습률만큼 이동하도록 구현하는 최적화 알고리즘.
#           wi ← wi − η(∂E / ∂wi), η : 학습률
#  2. Momentum
#  3. AdaGrad
#  4. ADAM
#  5. Adadelta
#  6. RMSprop

# ▣ 모델 훈련 및 평가
#  - 손실함수 : 교차 엔트로피, -tf.reduce_sum(y_ * tf.log(y_conv))
#  - 경사 감소법 : ADAM 알고리즘, tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

#  - 정확도 측정 : 측정치와 정답 레이블이 같은지 비교 후 해당 Bool 값을 실수로 변환후 그것들의 평균을 구한다.
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

import time
# 시작 시간 체크
stime = time.time()

for i in range(1000):
    batch = mnist.train.next_batch(100)  # 100 개씩 배치 수행
    if i%100 == 0:
        train_accracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' %(i, train_accracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # batch[0] : 입력 이미지에 대한 텐서, batch[1] : 정답 레이블 텐서

test_batch = mnist.test.next_batch(1000)
print('test accuracy %g' % sess.run(accuracy, feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))