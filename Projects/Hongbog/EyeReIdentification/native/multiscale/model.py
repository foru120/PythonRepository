from tensorflow.contrib import slim
from Projects.Hongbog.EyeReIdentification.native.multiscale.constants import *

class Layers:
    def __init__(self, is_training):
        self.is_training = is_training
        self.weights_initializers = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights_regularizers = tf.contrib.layers.l2_regularizer(scale=0.0005)

    def batch_norm(self, inputs, act=tf.nn.relu6, name='batch_norm_layer'):
        '''
            Batch Normalization
             - scale=True, scale factor(gamma) 를 사용
             - center=True, shift factor(beta) 를 사용
        '''
        with tf.variable_scope(name_or_scope=name):
            return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, center=True, scale=True, fused=True,
                                                updates_collections=tf.GraphKeys.UPDATE_OPS, activation_fn=act,
                                                is_training=self.is_training, scope='batch_norm')

    def conv2d(self, inputs, filters, kernel_size=1, strides=1, padding='same', act=tf.identity, name='conv2d_layer'):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=padding, activation=act,
                                kernel_initializer=self.weights_initializers,
                                bias_initializer=self.weights_initializers,
                                kernel_regularizer=self.weights_regularizers,
                                bias_regularizer=self.weights_regularizers,
                                name=name)

    def dense(self, inputs, units, act=tf.identity, name='dense_layer'):
        return tf.layers.dense(inputs=inputs, units=units, activation=act,
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

    def global_avg_pool2d(self, inputs, name):
        N, H, W, C = inputs.get_shape().as_list()

        with tf.variable_scope(name_or_scope=name):
            return tf.layers.average_pooling2d(inputs=inputs, pool_size=(H, W), strides=(H, W), padding='valid')

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

    def cross_input_neighborhood(self, inputs1, inputs2, name='cross_input_neighborhood'):
        with tf.variable_scope(name_or_scope=name):
            N, H, W, C = inputs1.get_shape().as_list()

            f = tf.div(inputs1, tf.norm(inputs1, axis=-1, keep_dims=True))
            g = tf.div(inputs2, tf.norm(inputs2, axis=-1, keep_dims=True))

            f = tf.transpose(f, [0, 3, 1, 2])
            m1 = tf.ones([N, C, H, W, 5, 5])
            f = tf.reshape(f, [N, C, H, W, 1, 1])
            f = tf.multiply(f, m1)

            g = tf.transpose(g, [0, 3, 1, 2])
            g = tf.reshape(g, [1, N, C, H, W])
            g_list = []
            g = tf.pad(g, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
            for i in range(H):
                for j in range(W):
                    g_list.append(g[:, :, :, i:i + 5, j:j + 5])

            concat = tf.concat(g_list, axis=0)
            g = tf.reshape(concat, [H, W, N, C, 5, 5])
            g = tf.transpose(g, [2, 3, 0, 1, 4, 5])

            diff1 = tf.reshape(tf.subtract(f, g), [N, C, H * 5, W * 5])
            diff2 = tf.reshape(tf.subtract(g, f), [N, C, H * 5, W * 5])
            diff1 = self.global_avg_pool2d(inputs=tf.transpose(diff1, [0, 2, 3, 1]), name='diff1_global_avg_pool')
            diff2 = self.global_avg_pool2d(inputs=tf.transpose(diff2, [0, 2, 3, 1]), name='diff2_global_avg_pool')
            diff1 = tf.squeeze(tf.pow(diff1, 2, name='diff1_pow'))
            diff2 = tf.squeeze(tf.pow(diff2, 2, name='diff2_pow'))
            diff1 = self.batch_norm(inputs=diff1, name='diff1_batch_norm')
            diff2 = self.batch_norm(inputs=diff2, name='diff2_batch_norm')
            diff1 = self.dropout(inputs=diff1, rate=0.4, name='diff1_dropout')
            diff2 = self.dropout(inputs=diff2, rate=0.4, name='diff2_dropout')

        return tf.concat([diff1, diff2], axis=-1, name='cin_output')

class Model(Layers):
    def __init__(self, sess, lr, batch_size, is_training, name):
        super(Model, self).__init__(is_training=is_training)
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.name = name
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(name_or_scope=self.name):
            with tf.variable_scope(name_or_scope='input_module'):
                self.ori_low_res_X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 46, 100, 1], name='ori_low_res_X')
                self.ori_mid_res_X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 70, 150, 1], name='ori_mid_res_X')
                self.ori_high_res_X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 92, 200, 1], name='ori_high_res_X')

                self.query_low_res_X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 46, 100, 1], name='query_low_res_X')
                self.query_mid_res_X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 70, 150, 1], name='query_mid_res_X')
                self.query_high_res_X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 92, 200, 1], name='query_high_res_X')

                self.low_res_X = tf.concat([self.ori_low_res_X, self.query_low_res_X], axis=0, name='low_res_X')
                self.mid_res_X = tf.concat([self.ori_mid_res_X, self.query_mid_res_X], axis=0, name='mid_res_X')
                self.high_res_X = tf.concat([self.ori_high_res_X, self.query_high_res_X], axis=0, name='high_res_X')

                self.y = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='Y')

            '''Low Resolution Image Network'''
            with tf.variable_scope(name_or_scope='low_res_module'):
                low_layer = self.conv2d(inputs=self.low_res_X, filters=32, kernel_size=3, strides=2, name='conv2d_0')
                low_layer = self.batch_norm(inputs=low_layer, name='conv2d_0_batch')

                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=32, strides=1, repeat=3, factor=6, name='bottleneck_3')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=96, strides=2, repeat=3, factor=6, name='bottleneck_5')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=140, strides=2, repeat=3, factor=6, name='bottleneck_6')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=200, strides=1, repeat=1, factor=6, name='bottleneck_7')
                low_layer = self.conv2d(inputs=low_layer, filters=300, name='conv2d_8')
                low_layer = self.batch_norm(inputs=low_layer, name='conv2d_8_batch')

                low_layer = tf.image.resize_nearest_neighbor(images=low_layer, size=(3, 7), name='resize')

            '''Mid Resolution Image Network'''
            with tf.variable_scope(name_or_scope='mid_res_module'):
                mid_layer = self.conv2d(inputs=self.mid_res_X, filters=32, kernel_size=3, strides=2, name='conv2d_0')
                mid_layer = self.batch_norm(inputs=mid_layer, name='conv2d_0_batch')

                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=32, strides=1, repeat=3, factor=6, name='bottleneck_3')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=96, strides=2, repeat=3, factor=6, name='bottleneck_5')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=140, strides=2, repeat=3, factor=6, name='bottleneck_6')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=200, strides=1, repeat=1, factor=6, name='bottleneck_7')
                mid_layer = self.conv2d(inputs=mid_layer, filters=300, name='conv2d_8')
                mid_layer = self.batch_norm(inputs=mid_layer, name='conv2d_8_batch')

                mid_layer = tf.image.resize_nearest_neighbor(images=mid_layer, size=(3, 7), name='resize')

            '''High Resolution Image Network'''
            with tf.variable_scope(name_or_scope='high_res_module'):
                high_layer = self.conv2d(inputs=self.high_res_X, filters=32, kernel_size=3, strides=2, name='conv2d_0')
                high_layer = self.batch_norm(inputs=high_layer, name='conv2d_0_batch')

                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=32, strides=1, repeat=3, factor=6, name='bottleneck_3')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=96, strides=2, repeat=3, factor=6, name='bottleneck_5')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=140, strides=2, repeat=3, factor=6, name='bottleneck_6')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=200, strides=1, repeat=1, factor=6, name='bottleneck_7')
                high_layer = self.conv2d(inputs=high_layer, filters=300, name='conv2d_8')
                high_layer = self.batch_norm(inputs=high_layer, name='conv2d_8_batch')

            with tf.variable_scope('multi_scale_module'):
                '''Low Resolution Feature Network'''
                low_ori_feature = tf.concat([low_layer[:self.batch_size], mid_layer[:self.batch_size], high_layer[:self.batch_size]], axis=-1, name='low_ori_feature')
                low_query_feature = tf.concat([low_layer[self.batch_size:], mid_layer[self.batch_size:], high_layer[self.batch_size:]], axis=-1, name='low_query_feature')

                low_res_cin = self.cross_input_neighborhood(inputs1=low_ori_feature, inputs2=low_query_feature, name='low_res_cin')

                '''Mid Resolution Feature Network'''
                mid_ori_feature = self.conv2d(inputs=low_ori_feature, filters=int(low_ori_feature.get_shape().as_list()[-1] / 2), name='mid_ori_feature_conv2d')
                mid_ori_feature = tf.layers.conv2d_transpose(inputs=mid_ori_feature, filters=mid_ori_feature.get_shape().as_list()[-1],
                                                             kernel_size=(3, 3), strides=(2, 2), padding='same', name='mid_ori_feature_resize')
                mid_query_feature = self.conv2d(inputs=low_query_feature, filters=int(low_query_feature.get_shape().as_list()[-1] / 2), name='mid_query_feature_conv2d')
                mid_query_feature = tf.layers.conv2d_transpose(inputs=mid_query_feature, filters=mid_query_feature.get_shape().as_list()[-1],
                                                               kernel_size=(3, 3), strides=(2, 2), padding='same', name='mid_query_feature_resize')

                mid_res_cin = self.cross_input_neighborhood(inputs1=mid_ori_feature, inputs2=mid_query_feature, name='mid_res_cin')

                '''High Resolution Feature Network'''
                high_ori_feature = self.conv2d(inputs=mid_ori_feature, filters=int(mid_ori_feature.get_shape().as_list()[-1] / 2), name='high_ori_feature_conv2d')
                high_ori_feature = tf.layers.conv2d_transpose(inputs=high_ori_feature, filters=high_ori_feature.get_shape().as_list()[-1],
                                                              kernel_size=(3, 3), strides=(2, 2), padding='same', name='high_ori_feature_resize')
                high_query_feature = self.conv2d(inputs=mid_query_feature, filters=int(mid_query_feature.get_shape().as_list()[-1] / 2), name='high_query_feature_conv2d')
                high_query_feature = tf.layers.conv2d_transpose(inputs=high_query_feature, filters=high_query_feature.get_shape().as_list()[-1],
                                                                kernel_size=(3, 3), strides=(2, 2), padding='same', name='high_query_feature_resize')

                high_res_cin = self.cross_input_neighborhood(inputs1=high_ori_feature, inputs2=high_query_feature, name='high_res_cin')

                tot_res_cin = tf.concat([low_res_cin, mid_res_cin, high_res_cin], axis=-1, name='tot_res_cin')

                self.logits = self.dense(inputs=tot_res_cin, units=2, name='logits')

            with tf.variable_scope('output_module'):
                self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

                self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
                self.loss = tf.add_n([self.loss] +
                                     [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float32))

                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

                update_ops = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
                with tf.control_dependencies(update_ops):
                    self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables)

    def train(self, ori_low_res_X, ori_mid_res_X, ori_high_res_X, query_low_res_X, query_mid_res_X, query_high_res_X, y):
        return self.sess.run([self.accuracy, self.loss, self.train_op],
                             feed_dict={self.ori_low_res_X: ori_low_res_X, self.ori_mid_res_X: ori_mid_res_X, self.ori_high_res_X: ori_high_res_X,
                                        self.query_low_res_X: query_low_res_X, self.query_mid_res_X: query_mid_res_X, self.query_high_res_X: query_high_res_X, self.y: y})

    def validation(self, ori_low_res_X, ori_mid_res_X, ori_high_res_X, query_low_res_X, query_mid_res_X, query_high_res_X, y):
        return self.sess.run([self.accuracy, self.loss, self.prob],
                             feed_dict={self.ori_low_res_X: ori_low_res_X, self.ori_mid_res_X: ori_mid_res_X, self.ori_high_res_X: ori_high_res_X,
                                        self.query_low_res_X: query_low_res_X, self.query_mid_res_X: query_mid_res_X, self.query_high_res_X: query_high_res_X, self.y: y})

    def predict(self, ori_low_res_X, ori_mid_res_X, ori_high_res_X, query_low_res_X, query_mid_res_X, query_high_res_X):
        return self.sess.run(self.prob,
                             feed_dict={self.ori_low_res_X: ori_low_res_X, self.ori_mid_res_X: ori_mid_res_X, self.ori_high_res_X: ori_high_res_X,
                                        self.query_low_res_X: query_low_res_X, self.query_mid_res_X: query_mid_res_X, self.query_high_res_X: query_high_res_X})