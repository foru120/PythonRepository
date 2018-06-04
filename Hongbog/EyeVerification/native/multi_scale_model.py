from Hongbog.EyeVerification.native.constants import *

import math

class Layers:
    weights_initializers = tf.contrib.layers.xavier_initializer(uniform=False)
    weights_regularizers = tf.contrib.layers.l2_regularizer(scale=flags.FLAGS.regularization_scale)

    def batch_norm_layer(self, inputs, act=tf.nn.elu, is_training=True, name='batch_norm_layer'):
        with tf.variable_scope(name_or_scope=name):
            return tf.cond(tf.cast(is_training, tf.bool),
                           true_fn=lambda: tf.contrib.layers.batch_norm(inputs=inputs, activation_fn=act, is_training=True, reuse=False,
                                                        updates_collections=tf.GraphKeys.UPDATE_OPS, scope='batch_norm'),
                           false_fn=lambda: tf.contrib.layers.batch_norm(inputs=inputs, activation_fn=act, is_training=False, reuse=True,
                                                        updates_collections=tf.GraphKeys.UPDATE_OPS, scope='batch_norm'))

    def conv2d_layer(self, inputs, filters, kernel_size, strides, padding, act, name='conv2d_layer'):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=padding, activation=act,
                                kernel_initializer=Layers.weights_initializers,
                                bias_initializer=Layers.weights_initializers,
                                kernel_regularizer=Layers.weights_regularizers,
                                bias_regularizer=Layers.weights_regularizers,
                                name=name)

    def residual_block(self, inputs, filters, kernel_size, is_training, name):
        with tf.variable_scope(name_or_scope=name):
            layer = self.batch_norm_layer(inputs=inputs, is_training=is_training, name='batch_norm_a')
            layer = self.conv2d_layer(inputs=layer, filters=filters, kernel_size=kernel_size, strides=(1, 1),
                                      padding='same', act=tf.identity, name='conv2d_a')
            layer = self.batch_norm_layer(inputs=layer, is_training=is_training, name='batch_norm_b')
            layer = self.conv2d_layer(inputs=layer, filters=filters, kernel_size=kernel_size, strides=(1, 1),
                                      padding='same', act=tf.identity, name='conv2d_b')
            layer = tf.add(inputs, layer, name='summation')
        return layer

    def dropout_layer(self, inputs, is_training, dropout_rate, name):
        with tf.variable_scope(name_or_scope=name):
            return tf.layers.dropout(inputs=inputs, rate=dropout_rate, training=is_training, name='dropout')

class Model(Layers):

    def __init__(self, sess, lr, name):
        self.sess = sess
        self.lr = lr
        self.name = name
        self._build_graph()

    def _build_graph(self):
        def _network():
            with tf.variable_scope(name_or_scope=self.name):
                with tf.variable_scope(name_or_scope='input_scope'):
                    self.low_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 60, 160, 1], name='low_res_X')
                    self.mid_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 80, 200, 1], name='mid_res_X')
                    self.high_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 100, 240, 1], name='high_res_X')
                    self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_data')

                    self.is_training = tf.placeholder(dtype=tf.bool, shape=None, name='is_training')
                    self.dropout_rate = tf.constant(0.4, dtype=tf.float32, shape=None, name='dropout_rate')

                '''Low Resolution Network'''
                with tf.variable_scope(name_or_scope='low_res_network'):
                    low_layer = self.conv2d_layer(inputs=self.low_res_X, filters=flags.FLAGS.hidden_num, kernel_size=(5, 5),
                                                  strides=(2, 2), padding='same', act=tf.identity, name='low_conv')

                    with tf.variable_scope('residual_network'):
                        for i in range(1, 20):
                            low_layer = self.residual_block(inputs=low_layer, filters=flags.FLAGS.hidden_num * math.ceil(i / 4),
                                                            kernel_size=(3, 3), is_training=self.is_training, name='residual_block_{}'.format(i))
                            if i % 4 == 0:
                                low_layer = self.dropout_layer(inputs=low_layer, is_training=self.is_training,
                                                               dropout_rate=self.dropout_rate,
                                                               name='low_res_dropout_{}'.format(int(i / 4)))
                                low_layer = self.conv2d_layer(inputs=low_layer,
                                                              filters=flags.FLAGS.hidden_num * (math.ceil(i / 4) + 1),
                                                              kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                              act=tf.identity, name='subsampling_{}'.format(int(i / 4)))
                    low_layer = tf.image.resize_nearest_neighbor(images=low_layer, size=(4, 8), name='resize')

                '''Mid Resolution Network'''
                with tf.variable_scope(name_or_scope='mid_res_network'):
                    mid_layer = self.conv2d_layer(inputs=self.mid_res_X, filters=flags.FLAGS.hidden_num, kernel_size=(5, 5),
                                                  strides=(2, 2), padding='same', act=tf.identity, name='mid_conv')

                    with tf.variable_scope('residual_network'):
                        for i in range(1, 20):
                            mid_layer = self.residual_block(inputs=mid_layer, filters=flags.FLAGS.hidden_num * math.ceil(i / 4),
                                                            kernel_size=(3, 3), is_training=self.is_training, name='residual_block_{}'.format(i))
                            if i % 4 == 0:
                                mid_layer = self.dropout_layer(inputs=mid_layer, is_training=self.is_training,
                                                               dropout_rate=self.dropout_rate,
                                                               name='mid_res_dropout_{}'.format(int(i / 4)))
                                mid_layer = self.conv2d_layer(inputs=mid_layer,
                                                              filters=flags.FLAGS.hidden_num * (math.ceil(i / 4) + 1),
                                                              kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                              act=tf.identity, name='subsampling_{}'.format(int(i / 4)))
                    mid_layer = tf.image.resize_nearest_neighbor(images=mid_layer, size=(4, 8), name='resize')

                '''High Resolution Network'''
                with tf.variable_scope(name_or_scope='high_res_network'):
                    high_layer = self.conv2d_layer(inputs=self.high_res_X, filters=flags.FLAGS.hidden_num, kernel_size=(5, 5),
                                                   strides=(2, 2), padding='same', act=tf.identity, name='high_conv')

                    with tf.variable_scope('residual_network'):
                        for i in range(1, 20):
                            high_layer = self.residual_block(inputs=high_layer, filters=flags.FLAGS.hidden_num * math.ceil(i / 4),
                                                             kernel_size=(3, 3), is_training=self.is_training, name='residual_block_{}'.format(i))
                            if i % 4 == 0:
                                high_layer = self.dropout_layer(inputs=high_layer, is_training=self.is_training,
                                                                dropout_rate=self.dropout_rate,
                                                                name='high_res_dropout_{}'.format(int(i / 4)))
                                high_layer = self.conv2d_layer(inputs=high_layer,
                                                               filters=flags.FLAGS.hidden_num * (math.ceil(i / 4) + 1),
                                                               kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                               act=tf.identity, name='subsampling_{}'.format(int(i / 4)))

                tot_layer = tf.concat([low_layer, mid_layer, high_layer], axis=-1, name='concat')
                self.cam_layer = tot_layer

                with tf.variable_scope('output_network'):
                    tot_layer = self.batch_norm_layer(inputs=tot_layer, name='batch_norm_output')
                    tot_layer = self.conv2d_layer(inputs=tot_layer, filters=7, kernel_size=(1, 1), strides=(1, 1),
                                                  padding='same', act=tf.identity, name='conv2d_output')
                    tot_layer = tf.layers.average_pooling2d(inputs=tot_layer, pool_size=(4, 8), strides=(1, 1), name='global_avg_pool')
                    tot_layer = tf.squeeze(input=tot_layer, axis=[1, 2], name='squeeze_output')

            self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

            return tot_layer

        self.logits = _network()

        with tf.variable_scope(name_or_scope=self.name):
            self.prob = tf.nn.softmax(logits=self.logits, name='softmax')
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
            self.loss = tf.add_n([self.loss] +
                                 [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float32))

            update_ops = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=self.variables)

    def train(self, low_res_X, mid_res_X, high_res_X, y, is_training):
        return self.sess.run([self.accuracy, self.loss, self.optimizer],
                             feed_dict={self.low_res_X: low_res_X, self.mid_res_X: mid_res_X, self.high_res_X: high_res_X,
                                        self.y: y, self.is_training: is_training})

    def validation(self, low_res_X, mid_res_X, high_res_X, y, is_training):
        return self.sess.run([self.accuracy, self.loss, self.prob],
                             feed_dict={self.low_res_X: low_res_X, self.mid_res_X: mid_res_X, self.high_res_X: high_res_X,
                                        self.y: y, self.is_training: is_training})