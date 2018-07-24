import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial)

def bias_variable(shape):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial_value=initial)

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable(shape[3])
    h = tf.nn.relu(conv2d(input, W) + b)
    hp = max_pool_2x2(h)
    return hp

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h1 = conv_layer(x_image, shape=[5, 5, 1, 32])
# W1 = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
# b1 = tf.constant(0.1, shape=[32])
# h1 = tf.nn.relu(tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
# hp1 = tf.nn.max_pool(value=h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
h2 = conv_layer(h1, shape=[5, 5, 32, 64])
# W2 = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
# b2 = tf.constant(0.1, shape=[32])
# h2 = tf.nn.relu(tf.nn.conv2d(hp1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
# hp2 = tf.nn.max_pool(value=h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
h3 = conv_layer(h2, shape=[5, 5, 64, 32])
# W3 = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
# b3 = tf.constant(0.1, shape=[32])
# h3 = tf.nn.relu(tf.nn.conv2d(hp2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)
# hp3 = tf.nn.max_pool(value=h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#todo 7.1.1 추상화 라이브러리 둘러보기
'''
    - tf.contrib.learn: 내부 구조로 들어가기 쉽고 저수준
    - TFLearn: 풍부한 기능을 갖추고 있고 다양한 유형의 최신 모델링에 필요한 많은 요소를 가지고 있음
    - TF-Slim: 주로 복잡한 합성곱 신경망을 쉽게 설계할 수 있도록 만들어졌으며 다양한 사전 학습된 모델을 제공해 자체적으로 학습하는 데 들어가는 비용의 부담을 덜 수 있음
    - Keras: Tensorflow, Theano 둘 다 지원
'''