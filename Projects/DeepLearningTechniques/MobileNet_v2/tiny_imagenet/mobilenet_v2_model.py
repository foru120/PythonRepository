import tensorflow as tf
import tensorflow.contrib.slim as slim

from Projects.DeepLearningTechniques.MobileNet_v2.tiny_imagenet.constants import *

class Model:

    def __init__(self, sess, is_training, is_tb_logging, name):
        self.sess = sess
        self.width = flags.FLAGS.image_width
        self.height = flags.FLAGS.image_height
        self.channel = flags.FLAGS.image_channel
        self.lr = flags.FLAGS.learning_rate
        self.dr = flags.FLAGS.learning_rate_decay
        self.is_training = is_training
        self.is_tb_logging = is_tb_logging
        self.name = name

        self.weights_initializers = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights_regularizers = tf.contrib.layers.l2_regularizer(scale=flags.FLAGS.l2_scale)

        self.summary_values = []
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name_or_scope='input_scope'):
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
                self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
                self.global_step = tf.Variable(0, trainable=False)

            with tf.variable_scope(name_or_scope='body_scope'):
                layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=1, name='conv2d_0')
                layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')

                layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
                layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
                layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=2, repeat=3, factor=6, name='bottleneck_3')
                layer = self.inverted_bottleneck(inputs=layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
                layer = self.inverted_bottleneck(inputs=layer, filters=96, strides=1, repeat=3, factor=6, name='bottleneck_5')
                layer = self.inverted_bottleneck(inputs=layer, filters=160, strides=2, repeat=3, factor=6, name='bottleneck_6')
                layer = self.inverted_bottleneck(inputs=layer, filters=320, strides=1, repeat=1, factor=6, name='bottleneck_7')

                if self.is_tb_logging:
                    self.summary_values.append(tf.summary.histogram('bottleneck_module', layer))

                layer = self.conv2d(inputs=layer, filters=1280, name='conv2d_8')
                layer = self.batch_norm(inputs=layer, name='conv2d_8_batch')
                self.cam_layer = layer
                layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='conv2d_8_dropout')
                layer = tf.layers.average_pooling2d(inputs=layer, pool_size=4, strides=1, name='conv2d_8_avg_pool')
                layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='conv2d_8_output')
                self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')

            with tf.variable_scope(name_or_scope='output_scope'):
                self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

                self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
                self.loss = tf.add_n([self.loss] +
                                     [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')

                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, -1), self.y), dtype=tf.float32))

                if self.is_tb_logging:
                    self.summary_values.append(tf.summary.scalar('loss', self.loss))
                    self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

                self.decay_lr = tf.train.exponential_decay(self.lr, self.global_step, 1000, flags.FLAGS.learning_rate_decay, staircase=True)

                # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.decay_lr)
                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.decay_lr)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.decay_lr)

                update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
                with tf.control_dependencies(update_opt):
                    self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

                if self.is_tb_logging:
                    self.summary_merged_values = tf.summary.merge(inputs=self.summary_values)

    def batch_norm(self, inputs, act=tf.nn.relu6, name='batch_norm_layer'):
        '''
            Batch Normalization
             - scale=True, scale factor(gamma) 를 사용
             - center=True, shift factor(beta) 를 사용
        '''
        with tf.variable_scope(name_or_scope=name):
            return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, center=True, scale=True, fused=True,
                                                updates_collections=tf.GraphKeys.UPDATE_OPS, activation_fn=act,
                                                zero_debias_moving_mean=True,
                                                is_training=self.is_training, scope='batch_norm')

    def conv2d(self, inputs, filters, kernel_size=1, strides=1, padding='same', act=tf.identity, name='conv2d_layer'):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=padding, activation=act,
                                kernel_initializer=self.weights_initializers,
                                bias_initializer=self.weights_initializers,
                                kernel_regularizer=self.weights_regularizers,
                                bias_regularizer=self.weights_regularizers,
                                name=name)

    def dropout(self, inputs, rate, name):
        with tf.variable_scope(name_or_scope=name):
            return tf.layers.dropout(inputs=inputs, rate=rate, training=self.is_training, name='dropout')

    def depthwise_conv2d(self, inputs, kernel_size=3, strides=2, padding='SAME', depth_multiplier=1, name=None):
        layer = slim.separable_conv2d(inputs=inputs, num_outputs=None, kernel_size=kernel_size, activation_fn=tf.identity,
                                      weights_initializer=self.weights_initializers, weights_regularizer=self.weights_regularizers,
                                      depth_multiplier=depth_multiplier, stride=strides, padding=padding, scope=name)
        return layer

    def inverted_bottleneck(self, inputs, filters, strides, repeat, factor, name=None):
        def _mobilenet_block(inputs, input_filters, output_filters, strides, name):
            with tf.variable_scope(name_or_scope=name):
                layer = self.conv2d(inputs=inputs, filters=input_filters * factor, name='bottleneck_layer')
                layer = self.batch_norm(inputs=layer, name='bottleneck_batch')

                layer = self.depthwise_conv2d(inputs=layer, strides=strides, name='depthwise_layer')
                layer = self.batch_norm(inputs=layer, name='depthwise_batch')

                layer = self.conv2d(inputs=layer, filters=output_filters, name='linear_layer')
                layer = self.batch_norm(inputs=layer, act=tf.identity, name='linear_batch')
            return layer

        prev_layer = inputs
        input_filters = inputs.get_shape().as_list()[-1]

        with tf.variable_scope(name_or_scope=name):
            for idx in range(repeat):
                layer = _mobilenet_block(inputs=prev_layer, input_filters=input_filters, output_filters=filters,
                                         strides=strides, name='mobilenet_block_{}'.format(idx))

                '''inverted_bottleneck 내의 첫 번째 layer 가 strides=2 인 경우 shortcut connection 생략'''
                if idx != 0 and strides != 2:
                    if prev_layer.get_shape().as_list()[-1] != layer.get_shape().as_list()[-1]:
                        prev_layer = self.conv2d(inputs=prev_layer, filters=filters, name='residual_match_{}'.format(idx))

                    layer = tf.add(prev_layer, layer, name='residual_add_{}'.format(idx))

                '''마지막 repeat 단계는 제외'''
                if idx != repeat-1:
                    strides = 1
                    prev_layer = layer

        return layer

    def train(self, x, y):
        if self.is_tb_logging:
            return self.sess.run([self.accuracy, self.loss, self.summary_merged_values, self.train_op], feed_dict={self.x: x, self.y: y})
        else:
            return self.sess.run([self.accuracy, self.loss, self.train_op], feed_dict={self.x: x, self.y: y})

    def validation(self, x, y):
        return self.sess.run([self.accuracy, self.loss, self.prob], feed_dict={self.x: x, self.y: y})

    def test(self, x):
        return self.sess.run(self.prob, feed_dict={self.x: x})