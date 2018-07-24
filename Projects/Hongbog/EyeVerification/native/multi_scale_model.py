from .constants import *

import math

class Layers:
    def __init__(self, is_training, is_logging):
        self.is_training = is_training
        self.is_logging = is_logging
        self.weights_initializers = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights_regularizers = tf.contrib.layers.l2_regularizer(scale=flags.FLAGS.regularization_scale)

    def batch_norm_layer(self, inputs, act=tf.nn.relu6, name='batch_norm_layer'):
        '''
            Batch Normalization
             - scale=True, scale factor(gamma) 를 사용
             - center=True, shift factor(beta) 를 사용
        '''
        with tf.variable_scope(name_or_scope=name):
            if self.is_training:
                return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, center=True, scale=True, fused=True,
                                                    updates_collections=tf.GraphKeys.UPDATE_OPS, activation_fn=act, is_training=True, scope='batch_norm')
            else:
                return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, center=True, scale=True, fused=True,
                                                    updates_collections=tf.GraphKeys.UPDATE_OPS, activation_fn=act, is_training=False, scope='batch_norm')

    def batch_norm_wrapper(self, inputs, decay=0.9, epsilon=1e-3, name='batch_norm_wrapper'):
        with tf.variable_scope(name_or_scope=name):
            gamma = tf.Variable(tf.ones(inputs.get_shape().as_list()[-1]), name='gamma')
            beta = tf.Variable(tf.zeros(inputs.get_shape().as_list()[-1]), name='beta')
            moving_mean = tf.Variable(tf.zeros(inputs.get_shape().as_list()[-1]), trainable=False, name='moving_mean')
            moving_var = tf.Variable(tf.ones(inputs.get_shape().as_list()[-1]), trainable=False, name='moving_var')

            if self.is_training:
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
                train_mean = tf.assign(moving_mean,
                                        moving_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(moving_var,
                                       moving_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,
                                                     batch_mean, batch_var, beta, gamma, epsilon)
            else:
                return tf.nn.batch_normalization(inputs,
                                                 moving_mean, moving_var, beta, gamma, epsilon)

    def group_norm_layer(self, inputs, G=32, epsilon=1e-5, name='group_norm'):
        '''
            G=1, Layer Normalization
            G=C, Instance Normalization
        '''
        with tf.variable_scope(name_or_scope=name):
            N, H, W, C = [-1 if shape == None else shape for shape in inputs.get_shape().as_list()]
            inputs = tf.reshape(tensor=inputs, shape=[N, G, H, W, C // G])

            mean, var = tf.nn.moments(x=inputs, axes=[2, 3, 4], keep_dims=True)
            inputs = (inputs - mean) / tf.sqrt(var + epsilon)

            inputs = tf.reshape(tensor=inputs, shape=[N, H, W, C])

            gamma = tf.Variable(initial_value=1., name='gamma')
            beta = tf.Variable(initial_value=0., name='beta')

            return gamma * inputs + beta

    def conv2d_layer(self, inputs, filters, kernel_size, strides, padding, act, name='conv2d_layer'):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=padding, activation=act,
                                kernel_initializer=self.weights_initializers,
                                bias_initializer=self.weights_initializers,
                                kernel_regularizer=self.weights_regularizers,
                                bias_regularizer=self.weights_regularizers,
                                name=name)

    def dropout_layer(self, inputs, rate, training, name):
        with tf.variable_scope(name_or_scope=name):
            if self.is_training:
                return tf.layers.dropout(inputs=inputs, rate=rate, training=True, name='dropout')
            else:
                return tf.layers.dropout(inputs=inputs, rate=0., training=False, name='dropout')

    def residual_block(self, inputs, filters, kernel_size, is_training, name):
        with tf.variable_scope(name_or_scope=name):
            # layer = self.batch_norm_wrapper(inputs=inputs, name='batch_norm_a')
            layer = self.batch_norm_layer(inputs=inputs, name='batch_norm_a')
            # layer = self.group_norm_layer(inputs=inputs, G=10, name='group_norm_a')
            # layer = tf.nn.relu6(layer, name='relu_a')
            layer = self.conv2d_layer(inputs=layer, filters=filters, kernel_size=kernel_size, strides=(1, 1),
                                      padding='same', act=tf.identity, name='conv2d_a')
            # layer = self.batch_norm_wrapper(inputs=layer, name='batch_norm_b')
            layer = self.batch_norm_layer(inputs=layer, name='batch_norm_b')
            # layer = self.group_norm_layer(inputs=layer, G=10, name='group_norm_b')
            # layer = tf.nn.relu6(layer, name='relu_b')
            layer = self.conv2d_layer(inputs=layer, filters=filters, kernel_size=kernel_size, strides=(1, 1),
                                      padding='same', act=tf.identity, name='conv2d_b')
            layer = tf.add(inputs, layer, name='summation')
        return layer

class Model(Layers):
    def __init__(self, sess, lr, is_training, is_logging, name):
        super(Model, self).__init__(is_training=is_training, is_logging=is_logging)
        self.sess = sess
        self.lr = lr
        self.name = name
        self.summary_values = []
        self._build_graph()

    def _build_graph(self):
        def _network():
            with tf.variable_scope(name_or_scope=self.name):
                with tf.variable_scope(name_or_scope='input_scope'):
                    self.low_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 60, 160, 1], name='low_res_X')
                    self.mid_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 80, 200, 1], name='mid_res_X')
                    self.high_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 100, 240, 1], name='high_res_X')
                    self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_data')

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
                                low_layer = self.conv2d_layer(inputs=low_layer,
                                                              filters=flags.FLAGS.hidden_num * (math.ceil(i / 4) + 1),
                                                              kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                              act=tf.identity, name='subsampling_{}'.format(int(i / 4)))
                                low_layer = self.dropout_layer(inputs=low_layer, rate=self.dropout_rate,
                                                               training=self.is_training,
                                                               name='low_res_dropout_{}'.format(int(i / 4)))
                        if self.is_logging:
                            self.summary_values.append(tf.summary.histogram('residual_network', low_layer))

                    # low_layer = tf.layers.conv2d_transpose(inputs=low_layer, filters=low_layer.get_shape()[-1],
                    #                                        kernel_size=(3, 4), strides=(1, 1), padding='VALID', name='resize')
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
                                mid_layer = self.conv2d_layer(inputs=mid_layer,
                                                              filters=flags.FLAGS.hidden_num * (math.ceil(i / 4) + 1),
                                                              kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                              act=tf.identity, name='subsampling_{}'.format(int(i / 4)))
                                mid_layer = self.dropout_layer(inputs=mid_layer, rate=self.dropout_rate,
                                                               training=self.is_training,
                                                               name='mid_res_dropout_{}'.format(int(i / 4)))
                        if self.is_logging:
                            self.summary_values.append(tf.summary.histogram('residual_network', mid_layer))

                    # mid_layer = tf.layers.conv2d_transpose(inputs=mid_layer, filters=mid_layer.get_shape()[-1],
                    #                                        kernel_size=(2, 2), strides=(1, 1), padding='VALID', name='resize')
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
                                high_layer = self.conv2d_layer(inputs=high_layer,
                                                               filters=flags.FLAGS.hidden_num * (math.ceil(i / 4) + 1),
                                                               kernel_size=(3, 3), strides=(2, 2), padding='same',
                                                               act=tf.identity, name='subsampling_{}'.format(int(i / 4)))
                                high_layer = self.dropout_layer(inputs=high_layer, rate=self.dropout_rate,
                                                                training=self.is_training,
                                                                name='high_res_dropout_{}'.format(int(i / 4)))
                        if self.is_logging:
                            self.summary_values.append(tf.summary.histogram('residual_network', high_layer))

                tot_layer = tf.concat([low_layer, mid_layer, high_layer], axis=-1, name='concat')
                self.cam_layer = tot_layer

                with tf.variable_scope('output_network'):
                    # tot_layer = self.batch_norm_wrapper(inputs=tot_layer, name='batch_norm_output')
                    # tot_layer = tf.nn.relu6(tot_layer, name='relu_output')
                    tot_layer = self.batch_norm_layer(inputs=tot_layer, name='batch_norm_output')
                    # tot_layer = self.group_norm_layer(inputs=tot_layer, G=10, name='group_norm_output')
                    if self.is_logging:
                        self.summary_values.append(tf.summary.histogram('output_network', tot_layer))
                    tot_layer = self.dropout_layer(inputs=tot_layer, rate=self.dropout_rate, training=self.is_training,
                                                   name='dropout_output')
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

            if self.is_logging:
                self.summary_values.append(tf.summary.scalar('loss', self.loss))
                self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            update_ops = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables)

            if self.is_logging:
                self.merged_values = tf.summary.merge(self.summary_values)

    def train(self, low_res_X, mid_res_X, high_res_X, y):
        if self.is_logging:
            return self.sess.run([self.accuracy, self.loss, self.merged_values, self.train_op],
                                  feed_dict={self.low_res_X: low_res_X, self.mid_res_X: mid_res_X, self.high_res_X: high_res_X,
                                             self.y: y})
        else:
            return self.sess.run([self.accuracy, self.loss, self.train_op],
                                 feed_dict={self.low_res_X: low_res_X, self.mid_res_X: mid_res_X,
                                            self.high_res_X: high_res_X, self.y: y})

    def validation(self, low_res_X, mid_res_X, high_res_X, y):
        if self.is_logging:
            return self.sess.run([self.accuracy, self.loss, self.prob, self.merged_values],
                                  feed_dict={self.low_res_X: low_res_X, self.mid_res_X: mid_res_X, self.high_res_X: high_res_X,
                                             self.y: y})
        else:
            return self.sess.run([self.accuracy, self.loss, self.prob],
                                 feed_dict={self.low_res_X: low_res_X, self.mid_res_X: mid_res_X,
                                            self.high_res_X: high_res_X, self.y: y})

    def predict(self, low_res_X, mid_res_X, high_res_X):
        return self.sess.run(self.prob,
                             feed_dict={self.low_res_X: low_res_X, self.mid_res_X: mid_res_X, self.high_res_X: high_res_X})