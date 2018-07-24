import tensorflow as tf
import tensorlayer as tl
import cv2
import numpy as np
from tensorlayer.layers import *

class Model:
    '''
        Inception-resnet-v2 + Densenet 기반의 신경망
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
            self.dropout_rate = 0.5
            self.learning_rate = 0.1
            self.regularizer = tf.contrib.layers.l2_regularizer(0.001)

        def _batch_norm(layer, act=tf.nn.elu, name='batch_norm'):
            with tf.variable_scope(name_or_scope=name):
                return BatchNormLayer(layer, act=act, is_train=self.training)

        def _dense_block(input, name):
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

        def _transition(layer, name):
            with tf.variable_scope(name):
                '''compression transition layer (DenseNet-C)'''
                layer = _batch_norm(layer=layer, name='transition_batch_norm')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, is_fix=True, name='transition_dropout_a')
                layer = Conv2d(layer=layer, n_filter=layer.outputs.get_shape()[-1], filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='transition_conv')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='transition_pool')
            return layer

        def _inception_resnet_a(input, name):
            with tf.variable_scope(name):
                sub_layer_a = Conv2d(layer=input, n_filter=32, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a01')
                sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=48, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a02')
                sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a03')

                sub_layer_b = Conv2d(layer=input, n_filter=32, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_b01')
                sub_layer_b = Conv2d(layer=sub_layer_b, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='sub_layer_b02')

                sub_layer_c = Conv2d(layer=input, n_filter=32, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_c01')

                layer = ConcatLayer([sub_layer_a, sub_layer_b, sub_layer_c], name='concat_layer_01')
                layer = Conv2d(layer=layer, n_filter=384, filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='output_layer')
                layer.outputs = input.outputs + layer.outputs
            return layer

        def _inception_resnet_b(input, name):
            with tf.variable_scope(name):
                # batch_input = _batch_norm(input, act=tf.nn.elu, name='batch_norm_input')
                sub_layer_a = Conv2d(layer=input, n_filter=32, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a01')
                # sub_layer_a = _batch_norm(layer=sub_layer_a, act=tf.nn.elu, name='batch_norm_a01')
                sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=48, filter_size=(1, 5), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a02')
                # sub_layer_a = _batch_norm(layer=sub_layer_a, act=tf.nn.elu, name='batch_norm_a02')
                sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=64, filter_size=(5, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a03')
                # sub_layer_a = _batch_norm(layer=sub_layer_a, act=tf.nn.elu, name='batch_norm_a03')

                sub_layer_b = Conv2d(layer=input, n_filter=64, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_b01')
                # sub_layer_b = _batch_norm(layer=sub_layer_b, act=tf.nn.elu, name='batch_norm_b01')

                layer = ConcatLayer([sub_layer_a, sub_layer_b], name='concat_layer_01')
                layer = Conv2d(layer=layer, n_filter=128, filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='output_layer')
                layer.outputs = input.outputs + layer.outputs
            return layer

        def _inception_resnet_c(input, name):
            with tf.variable_scope(name):
                # batch_input = _batch_norm(input, act=tf.nn.elu, name='batch_norm_input')
                sub_layer_a = Conv2d(layer=input, n_filter=64, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a01')
                # sub_layer_a = _batch_norm(layer=sub_layer_a, act=tf.nn.elu, name='batch_norm_a01')
                sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=96, filter_size=(1, 3), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a02')
                # sub_layer_a = _batch_norm(layer=sub_layer_a, act=tf.nn.elu, name='batch_norm_a02')
                sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=128, filter_size=(3, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a03')
                # sub_layer_a = _batch_norm(layer=sub_layer_a, act=tf.nn.elu, name='batch_norm_a03')

                sub_layer_b = Conv2d(layer=input, n_filter=64, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_b01')
                # sub_layer_b = _batch_norm(layer=sub_layer_b, act=tf.nn.elu, name='batch_norm_b01')

                layer = ConcatLayer([sub_layer_a, sub_layer_b], name='concat_layer_01')
                layer = Conv2d(layer=layer, n_filter=512, filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='output_layer')
                layer.outputs = input.outputs + layer.outputs
            return layer

        def _reduction_b(input, name):
            with tf.variable_scope(name):
                # batch_input = _batch_norm(input, act=tf.nn.elu, name='batch_norm_input')
                sub_layer_a = Conv2d(layer=input, n_filter=48, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a01')
                # sub_layer_a = _batch_norm(layer=sub_layer_a, act=tf.nn.elu, name='batch_norm_a01')
                sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=72, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a02')
                # sub_layer_a = _batch_norm(layer=sub_layer_a, act=tf.nn.elu, name='batch_norm_a02')
                sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=128, filter_size=(3, 3), strides=(2, 2), act=tf.nn.elu, name='sub_layer_a03')

                sub_layer_b = Conv2d(layer=input, n_filter=48, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_b01')
                # sub_layer_b = _batch_norm(layer=sub_layer_b, act=tf.nn.elu, name='batch_norm_b01')
                sub_layer_b = Conv2d(layer=sub_layer_b, n_filter=96, filter_size=(3, 3), strides=(2, 2), act=tf.nn.elu, name='sub_layer_b02')

                sub_layer_c = Conv2d(layer=input, n_filter=48, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_c01')
                # sub_layer_c = _batch_norm(layer=sub_layer_c, act=tf.nn.elu, name='batch_norm_c01')
                sub_layer_c = Conv2d(layer=sub_layer_c, n_filter=160, filter_size=(3, 3), strides=(2, 2), act=tf.nn.elu, name='sub_layer_c02')

                sub_layer_d = PoolLayer(prev_layer=input, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='sub_layer_d01')

                layer = ConcatLayer([sub_layer_a, sub_layer_b, sub_layer_c, sub_layer_d], name='concat_layer_01')
            return layer

        def _network():
            with tf.variable_scope(name_or_scope=self.name):
                with tf.variable_scope('stem_layer'):
                    # stem-layer part 1
                    layer = InputLayer(inputs=self.X, name='input')
                    layer = Conv2d(layer=layer, n_filter=16, filter_size=(3, 3), strides=(2, 2), act=tf.nn.elu, padding='VALID', name='conv_01')  # (49, 24)
                    layer = Conv2d(layer=layer, n_filter=16, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv_02')  # (49, 24)
                    layer = Conv2d(layer=layer, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv_03')  # (49, 24)

                    sub_layer_a = Conv2d(layer=layer, n_filter=64, filter_size=(3, 3), strides=(2, 2), act=tf.nn.elu, name='sub_layer_a01')  # (25, 12)
                    sub_layer_b = PoolLayer(prev_layer=layer, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='sub_layer_b01')  # (25, 12)

                    layer = ConcatLayer([sub_layer_a, sub_layer_b], name='concat_layer_01')

                    # stem-layer part 2
                    sub_layer_a = Conv2d(layer=layer, n_filter=32, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a02')
                    sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=32, filter_size=(5, 1), strides=(1 ,1), act=tf.nn.elu, name='sub_layer_a03')
                    sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=32, filter_size=(1, 5), strides=(1, 1), act=tf.nn.elu, name='sub_layer_a04')
                    sub_layer_a = Conv2d(layer=sub_layer_a, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, padding='VALID', name='sub_layer_a05')

                    sub_layer_b = Conv2d(layer=layer, n_filter=32, filter_size=(1, 1), strides=(1, 1), act=tf.nn.elu, name='sub_layer_b02')
                    sub_layer_b = Conv2d(layer=sub_layer_b, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, padding='VALID', name='sub_layer_b03')

                    layer = ConcatLayer([sub_layer_a, sub_layer_b], name='concat_layer_02')  # (23, 10, 128)

                    # sub_layer_a = PoolLayer(prev_layer=layer, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), pool=tf.nn.max_pool, name='sub_layer_a06')  # (12, 5)
                    # sub_layer_b = Conv2d(layer=layer, n_filter=128, filter_size=(3, 3), strides=(2, 2), name='sub_layer_b04')
                    # layer = ConcatLayer([sub_layer_a, sub_layer_b], name='concat_layer_03')

                # inception-resnet-B (6x4), input: (23, 10, 128), output: (23, 10, 128)
                with tf.variable_scope('inception_resnet_B'):
                    for i in range(6):
                        layer = _inception_resnet_b(input=layer, name='inception_resnet_B_{}'.format(i))

                # reduction-B (6x4), input: (23, 10, 128), output: (12, 5, 512)
                layer = _reduction_b(input=layer, name='reduction_B')

                # inception-resnet-C (3x4), input: (12, 5, 512), output: (12, 5, 512)
                with tf.variable_scope('inception_resnet_C'):
                    for i in range(3):
                        layer = _inception_resnet_c(input=layer, name='inception_resnet_C_{}'.format(i))

                # dense block, input: (12, 5, 512), output: (6, 3, 992)
                with tf.variable_scope('dense_block_01'):
                    for i in range(self.N):
                        layer = _dense_block(input=layer, name='dense_layer_{}'.format(i))
                    layer = _transition(layer, 'transition1')

                self.cam_layer = layer

                with tf.variable_scope('output_layer'):
                    # layer = _batch_norm(layer=layer, act=tf.nn.elu, name='batch_norm_output')
                    layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, is_fix=True, name='dropout_output')
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