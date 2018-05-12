import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import cv2

class Model:
    '''
        ResNet 기반의 신경망
    '''
    def __init__(self, sess, name, training):
        self.sess = sess
        self.name = name
        self.N = 12  # Dense Block 내의 Layer 개수
        self.growthRate = 10  # k
        self.compression_factor = 0.5
        self.hidden_num = 16
        self.training = training
        self._build_graph()

    def _build_graph(self):
        with tf.name_scope('initialize_scope'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 100, 50, 1], name='X_data')
            self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_data')
            self.dropout_rate = 0.6
            self.learning_rate = 0.001
            self.regularizer = tf.contrib.layers.l2_regularizer(0.001)

        def _batch_norm(layer, act=tf.nn.elu, name='batch_norm'):
            with tf.variable_scope(name_or_scope=name):
                return BatchNormLayer(layer, act=act, is_train=self.training)

        def _add_layer(input, name):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                layer = _batch_norm(layer=input, name='bottleneck_batch_norm')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, is_fix=True, name='bottleneck_dropout')
                layer = Conv2d(layer=layer, n_filter=4 * self.growthRate, filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='bottleneck_conv')

                '''basic dense layer'''
                layer = _batch_norm(layer=layer, name='basic_batch_norm')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, is_fix=True, name='basic_dropout_a')
                layer = DepthwiseConv2d(prev_layer=layer, shape=(3, 3), strides=(1, 1), act=tf.nn.elu, name='basic_dwconv')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, is_fix=True, name='basic_dropout_b')

                layer.outputs = tf.concat([layer.outputs, input.outputs], axis=3)
            return layer

        def _add_transition(layer, name):
            with tf.variable_scope(name):
                '''compression transition layer (DenseNet-C)'''
                layer = _batch_norm(layer=layer, name='transition_batch_norm')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, is_fix=True, name='transition_dropout_a')
                layer = Conv2d(layer=layer, n_filter=layer.outputs.get_shape()[-1], filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='transition_conv')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='transition_pool')
            return layer

        def _residual_block(input, n_filter, filter_size, strides, name):
            '''
            Improved ResNet
             - original resnet 에서 마지막 단계의 summation 연산 시 ReLU 함수를 사용한 것과 달리 identity 함수를 사용해
               identity mapping 이 이루어지게 된다.
             - full pre-activation: batch normalization 을 residual net 상에서 activation 함수보다 앞으로 오는 구조.
            '''
            with tf.variable_scope(name):
                layer = _batch_norm(input, act=tf.nn.elu, name='batch_norm_f1')
                layer = Conv2d(layer=layer, n_filter=n_filter, filter_size=filter_size, strides=strides, act=tf.identity, name='residual_f1')
                layer = _batch_norm(layer, act=tf.nn.elu, name='batch_norm_f2')
                layer = Conv2d(layer=layer, n_filter=n_filter, filter_size=filter_size, strides=strides, act=tf.identity, name='residual_f2')
                layer.outputs = input.outputs + layer.outputs
            return layer

        def _network():
            with tf.variable_scope(name_or_scope=self.name):
                with tf.variable_scope('input_layer'):
                    layer = InputLayer(inputs=self.X, name='input')
                    layer = Conv2d(layer=layer, n_filter=self.hidden_num, filter_size=(5, 5), strides=(2, 2), name='conv_01')  # 50, 25

                '''residual block'''
                with tf.variable_scope('residual_block'):
                    for i in range(1, 20):
                        layer = _residual_block(input=layer, n_filter=self.hidden_num, filter_size=(3, 3), strides=(1, 1), name='residual_layer_{}'.format(i))
                        if i % 4 == 0:
                            self.hidden_num *= 2
                            layer = Conv2d(layer=layer, n_filter=self.hidden_num, filter_size=(3, 3), strides=(2, 2), name='subsampling_{}'.format(int(i/4)))

                # '''dense block'''
                # with tf.variable_scope('dense_block_01'):
                #     for i in range(self.N):
                #         layer = _add_layer(input=layer, name='dense_layer_{}'.format(i))
                #     layer = _add_transition(layer, 'transition1')
                #
                # with tf.variable_scope('dense_block_02'):
                #     for i in range(self.N):
                #         layer = _add_layer(input=layer, name='dense_layer_{}'.format(i))

                self.cam_layer = layer

                with tf.variable_scope('output_layer'):
                    layer = Conv2d(layer=layer, n_filter=7, filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='logit')
                    layer = GlobalMeanPool2d(prev_layer=layer, name='global_avg_pool')

            return layer

        self.logits = _network()
        self.prob = tf.nn.softmax(logits=self.logits.outputs, name='softmax')
        self.loss = tl.cost.cross_entropy(output=self.logits.outputs, target=self.y, name='ce_loss')
        self.loss = self.loss + self.regularizer(self.logits.all_params[0]) + self.regularizer(self.logits.all_params[2])

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits.outputs, 1), self.y), dtype=tf.float64))

    def grad_cam(self, x_test, batch):
        '''
         Gradient를 통해 신경망의 판단을 역추적하여 가시화적인 이미지를 생성하는 기법. CAM의 일반적 기법.
         ∴ L(Grad-cam)^c = ReLU(∑a_k^c * A^k)
            c = 타겟 클래스
            a_k^c = ∑∑ ∂y^c / ∂A^k
            A^k = 마지막 단의 conv-layer의 k번째 피쳐맵.
            y^c = 소프트 맥스 레이어의 입력값.
         :param x_test: 입력 영상에 해당하는 파라미터.
         :param batch: 입력 영상의 batch-size.
         :return: gradient를 통해 계산된 ClassActivationMap이 생성됨.
         '''
        self.training = False
        self.dropout_rate = 1.0
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        tot_cam_list = []
        conv_layer = self.cam_layer.outputs
        # conv_layer = tf.get_variable(name=self.name + '/dense_block_02/dense_layer_11/basic_dwconv/W_depthwise2d') # 보고자 하는 layer 변수.
        # print('conv_layer.shape = ', conv_layer.shape) # conv_layer.shape = (?, N, N, Channel)
        signal = tf.multiply(self.logits.outputs, self.prob)  # self.logits -> softmax 거치기 전 변수, self.prob -> softmax 거친 변수
        # print('signal.shape = ', signal.shape) # (?, label_num)
        loss = tf.reduce_mean(signal)
        # tf.gradients(y, x) ~ ∂y / ∂x (편미분을 하려는 대상 / 편미분 변수)
        grads = tf.gradients(loss, conv_layer)[0]  # (?, N, N, Channel)

        for idx in range(batch):
            norm_grads = tf.div(grads[idx], tf.sqrt(tf.reduce_mean(tf.square(grads[idx]))) + tf.constant(1e-5))
            output, grads_val = self.sess.run([conv_layer[idx], norm_grads], feed_dict={self.X: x_test})

            weights = np.mean(grads_val, axis=(0, 1))
            cam = np.ones(output.shape[0: 2], dtype=np.float32)  # (N, N)

            for i, w in enumerate(weights):
                cam += w * output[:, :, i]

            # ReLU 씌우기.
            cam = np.maximum(cam, 0)
            cam = cam / np.max(cam)
            cam3 = cv2.resize(cam, (100, 50))
            tot_cam_list.append(cam3)
        return tot_cam_list

    def predict(self, x_test):
        self.training = False
        return self.sess.run(self.prob, feed_dict={self.X: x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.y: y_test})

    def train(self, x_data, y_data):
        self.training = True
        return self.sess.run([self.accuracy, self.loss, self.optimizer], feed_dict={self.X: x_data, self.y: y_data})

    def validation(self, x_test, y_test):
        self.training = False
        return self.sess.run([self.accuracy, self.loss, self.prob], feed_dict={self.X: x_test, self.y: y_test})