from keras.layers import Input, BatchNormalization, LeakyReLU, Add, Dropout, Concatenate, GlobalAveragePooling2D, Softmax
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras import initializers, regularizers
import tensorflow as tf
import math

class Layers:
    def __init__(self):
        self.kernel_initializer = initializers.he_normal(5)
        self.kernel_regularizer = regularizers.l2(1e-5)

    def conv2d(self, inputs, filters, kernel_size=1, strides=1, padding='same', act='linear', name='conv2d'):
        with tf.variable_scope(name_or_scope=name):
            return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=act,
                          kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)(inputs)

    def conv2d_transpose(self, inputs, filters, kernel_size, strides=1, padding='same', act='linear', name='conv2d_transpose'):
        with tf.variable_scope(name_or_scope=name):
            return Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=act,
                                   kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)(inputs)

    def dropout(self, inputs, rate, name):
        with tf.variable_scope(name_or_scope=name):
            return Dropout(rate=rate)(inputs)

    def global_avg_pool_2d(self, inputs, name='avg_pool_2d'):
        with tf.variable_scope(name_or_scope=name):
            return GlobalAveragePooling2D()(inputs)

    def batch_norm(self, inputs, is_act=True, name='batch_norm'):
        with tf.variable_scope(name_or_scope=name):
            layer = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3, center=True, scale=True)(inputs)
            if is_act:
                layer = LeakyReLU(alpha=0.3)(layer)
        return layer

    def residual_block(self, inputs, filters, kernel_size, name):
        with tf.variable_scope(name_or_scope=name):
            layer = self.batch_norm(inputs=inputs, is_act=True, name='batch_norm_a')
            layer = self.conv2d(inputs=layer, filters=filters, kernel_size=kernel_size, name='conv2d_a')
            layer = self.batch_norm(inputs=layer, is_act=True, name='batch_norm_b')
            layer = self.conv2d(inputs=layer, filters=filters, kernel_size=kernel_size, name='conv2d_b')
            layer = Add()([inputs, layer])
        return layer

class NeuralNet(Layers):
    def __init__(self, name):
        super(NeuralNet, self).__init__()
        self.name = name
        self.hidden_num = 30
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(name_or_scope=self.name):
            with tf.variable_scope(name_or_scope='input_module'):
                self.low_res_X = Input(shape=(60, 160, 1), dtype='float32', name='low_res_X')
                self.mid_res_X = Input(shape=(80, 200, 1), dtype='float32', name='mid_res_X')
                self.high_res_X = Input(shape=(100, 240, 1), dtype='float32', name='high_res_X')

            '''Low Resolution Network'''
            with tf.variable_scope(name_or_scope='low_res_module'):
                low_layer = self.conv2d(inputs=self.low_res_X, filters=self.hidden_num, kernel_size=3, strides=2, name='low_conv')

                with tf.variable_scope('residual_network'):
                    for i in range(1, 20):
                        low_layer = self.residual_block(inputs=low_layer, filters=self.hidden_num * math.ceil(i / 4),
                                                        kernel_size=3, name='residual_block_{}'.format(i))
                        if i % 4 == 0:
                            low_layer = self.conv2d(inputs=low_layer,
                                                    filters=self.hidden_num * (math.ceil(i / 4) + 1),
                                                    kernel_size=3, strides=2, name='subsampling_{}'.format(int(i / 4)))
                            low_layer = self.dropout(inputs=low_layer, rate=0.4, name='low_res_dropout_{}'.format(int(i / 4)))

                low_layer = self.conv2d_transpose(inputs=low_layer, filters=150, kernel_size=(3, 4), padding='valid', name='resize')

            '''Mid Resolution Network'''
            with tf.variable_scope(name_or_scope='mid_res_module'):
                mid_layer = self.conv2d(inputs=self.mid_res_X, filters=self.hidden_num, kernel_size=5, strides=2, name='mid_conv')

                with tf.variable_scope(name_or_scope='residual_network'):
                    for i in range(1, 20):
                        mid_layer = self.residual_block(inputs=mid_layer, filters=self.hidden_num * math.ceil(i / 4),
                                                        kernel_size=3, name='residual_block_{}'.format(i))
                        if i % 4 == 0:
                            mid_layer = self.conv2d(inputs=mid_layer, filters=self.hidden_num * (math.ceil(i / 4) + 1),
                                                    kernel_size=3, strides=2, name='subsampling_{}'.format(int(i / 4)))
                            mid_layer = self.dropout(inputs=mid_layer, rate=0.4, name='mid_res_dropout_{}'.format(int(i / 4)))

                mid_layer = self.conv2d_transpose(inputs=mid_layer, filters=150, kernel_size=(2, 2), padding='valid', name='resize')


            '''High Resolution Network'''
            with tf.variable_scope(name_or_scope='high_res_module'):
                high_layer = self.conv2d(inputs=self.high_res_X, filters=self.hidden_num, kernel_size=5, strides=2, name='high_conv')

                with tf.variable_scope(name_or_scope='residual_network'):
                    for i in range(1, 20):
                        high_layer = self.residual_block(inputs=high_layer, filters=self.hidden_num * math.ceil(i / 4),
                                                         kernel_size=3, name='residual_block_{}'.format(i))
                        if i % 4 == 0:
                            high_layer = self.conv2d(inputs=high_layer, filters=self.hidden_num * (math.ceil(i / 4) + 1),
                                                     kernel_size=3, strides=2, name='subsampling_{}'.format(int(i / 4)))
                            high_layer = self.dropout(inputs=high_layer, rate=0.4, name='high_res_dropout_{}'.format(int(i / 4)))

            tot_layer = Concatenate(axis=-1)([low_layer, mid_layer, high_layer])

            with tf.variable_scope(name_or_scope='output_module'):
                tot_layer = self.batch_norm(inputs=tot_layer, is_act=True, name='batch_norm_output')
                tot_layer = self.dropout(inputs=tot_layer, rate=0.4, name='dropout_output')
                tot_layer = self.conv2d(inputs=tot_layer, filters=7, kernel_size=1, name='conv2d_output')
                self.logits = self.global_avg_pool_2d(inputs=tot_layer, name='global_avg_pool')
                self.prob = Softmax(axis=-1, name='prob')(self.logits)