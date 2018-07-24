import tensorflow as tf
from tensorflow.contrib import slim

net = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=[11, 11], stride=4, padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0007), scope='conv1')

#todo tf-slim, repeat 을 통한 중복 코드 제거
net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], scope='conv1_1')
net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], scope='conv1_2')
net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], scope='conv1_3')
net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], scope='conv1_4')
net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], scope='conv1_5')

'''
    위 다섯 줄은 다음 한 줄로 대체 (계층의 크기가 같을 경우에만 사용 가능)
'''
net = slim.repeat(inputs=net, repetitions=5, layer=slim.conv2d, num_outputs=128, kernel_size=[3, 3], scope='conv1')

#todo tf-slim, stack 을 통한 계층 연결
net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], scope='conv1_1')
net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[1, 1], scope='conv1_2')
net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], scope='conv1_3')
net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[1, 1], scope='conv1_4')
net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], scope='conv1_5')

'''
    위 다섯 줄은 다음과 같이 변경 가능
'''
net = slim.stack(inputs=net, layer=slim.conv2d, [(64, [3, 3]), (64, [1, 1]), (128, [3, 3]), (128, [1, 1]), (256, [3, 3])], scope='conv')

#todo tf-slim, arg_scope 를 통한 공유 인수 집합 설정
with slim.arg_scope([slim.conv2d],
                    padding='VALID',
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    weights_regularizer=slim.l2_regularizer(0.0007)):
    net = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=[11, 11], scope='conv1')
    net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[11, 11], padding='VALID', scope='conv2')
    net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[11, 11], scope='conv3')
    net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[11, 11], scope='conv4')

#todo tf-slim, VGG 모델 구현
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs=inputs, repetitions=2, layer=slim.conv2d, num_outputs=64, kernel_size=[3, 3], scope='con1')
    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
    net = slim.repeat(inputs=net, repetitions=2, layer=slim.conv2d, num_outputs=128, kernel_size=[3, 3], scope='con2')
    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
    net = slim.repeat(inputs=net, repetitions=3, layer=slim.conv2d, num_outputs=128, kernel_size=[3, 3], scope='con3')
    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool3')
    net = slim.repeat(inputs=net, repetitions=3, layer=slim.conv2d, num_outputs=128, kernel_size=[3, 3], scope='con4')
    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool4')
    net = slim.repeat(inputs=net, repetitions=3, layer=slim.conv2d, num_outputs=128, kernel_size=[3, 3], scope='con5')
    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool5')
    net = slim.fully_connected(inputs=net, num_outputs=4096, scope='fc6')
    net = slim.dropout(inputs=net, keep_prob=0.5, scope='dropout6')
    net = slim.fully_connected(inputs=net, num_outputs=4096, scope='fc7')
    net = slim.dropout(inputs=net, keep_prob=0.5, scope='dropout7')
    net = slim.fully_connected(inputs=net, num_outputs=1000, activation_fn=None, scope='fc8')