from Hongbog.EyeVerification.native_v4.constants import *
import tensorflow.contrib.slim as slim

class Layers:
    def __init__(self, is_training, is_logging):
        self.is_training = is_training
        self.is_logging = is_logging
        self.weights_initializers = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights_regularizers = tf.contrib.layers.l2_regularizer(scale=flags.FLAGS.regularization_scale)

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

    def fully_connected(self, inputs, units, act, name='fully_connected_layer'):
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

    def squeeze_excitation(self, inputs, num_outputs, ratio=16, name='squeeze_excitation'):
        with tf.variable_scope(name):
            layer = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keep_dims=True, name='global_pool')
            layer = self.fully_connected(inputs=layer, units=num_outputs / ratio, act=tf.nn.relu6, name='dense_bottleneck')
            layer = self.fully_connected(inputs=layer, units=num_outputs, act=tf.nn.sigmoid, name='dense_expantion')
            layer = tf.reshape(layer, [-1, 1, 1, num_outputs])
        return tf.multiply(inputs, layer)

    def inception_resnet_A(self, inputs, filters, name):
        min_filters, median_filters, max_filters= int(filters/8), int(filters/4), int(filters/4) - int(filters/8)

        with tf.variable_scope(name):
            sub_layer_a = self.conv2d(inputs=inputs, filters=min_filters, act=tf.nn.relu6, name='sub_layer_a01')
            sub_layer_a = self.conv2d(inputs=sub_layer_a, filters=median_filters, kernel_size=3, act=tf.nn.relu6, name='sub_layer_a02')
            sub_layer_a = self.conv2d(inputs=sub_layer_a, filters=max_filters, kernel_size=3, act=tf.nn.relu6, name='sub_layer_a03')

            sub_layer_b = self.conv2d(inputs=inputs, filters=min_filters, act=tf.nn.relu6, name='sub_layer_b01')
            sub_layer_b = self.conv2d(inputs=sub_layer_b, filters=min_filters, kernel_size=3, act=tf.nn.relu6, name='sub_layer_b02')

            sub_layer_c = self.conv2d(inputs=inputs, filters=min_filters, act=tf.nn.relu6, name='sub_layer_c01')

            layer = tf.concat([sub_layer_a, sub_layer_b, sub_layer_c], axis=-1, name='concat_layer')
            layer = self.batch_norm(inputs=layer, name='concat_batch')

            layer = self.conv2d(inputs=layer, filters=filters, name='linear_layer')
            layer = self.batch_norm(inputs=layer, name='linear_batch')

            layer = tf.add(inputs, layer)
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

class Model(Layers):
    def __init__(self, sess, lr, is_training, is_logging, name):
        super(Model, self).__init__(is_training=is_training, is_logging=is_logging)
        self.sess = sess
        self.lr = lr
        self.name = name
        self.summary_values = []
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(name_or_scope=self.name):
            with tf.variable_scope(name_or_scope='input_module'):
                self.low_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 60, 160, 1], name='low_res_X')
                self.mid_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 80, 200, 1], name='mid_res_X')
                self.high_res_X = tf.placeholder(dtype=tf.float32, shape=[None, 100, 240, 1], name='high_res_X')
                self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_data')

            '''Low Resolution Network'''
            with tf.variable_scope(name_or_scope='low_res_module'):
                low_layer = self.conv2d(inputs=self.low_res_X, filters=32, kernel_size=3, strides=2, name='conv2d_0')
                low_layer = self.batch_norm(inputs=low_layer, name='conv2d_0_batch')

                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=32, strides=2, repeat=3, factor=6, name='bottleneck_3')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
                low_layer = self.inverted_bottleneck(inputs=low_layer, filters=96, strides=1, repeat=3, factor=6, name='bottleneck_5')
                low_layer = self.conv2d(inputs=low_layer, filters=320, name='conv2d_6')
                low_layer = self.batch_norm(inputs=low_layer, name='conv2d_6_batch')

                if self.is_logging:
                    self.summary_values.append(tf.summary.histogram('residual_network', low_layer))

                # low_layer = tf.layers.conv2d_transpose(inputs=low_layer, filters=low_layer.get_shape()[-1],
                #                                        kernel_size=(3, 4), strides=(1, 1), padding='VALID', name='resize')
                low_layer = tf.image.resize_nearest_neighbor(images=low_layer, size=(7, 15), name='resize')

            '''Mid Resolution Network'''
            with tf.variable_scope(name_or_scope='mid_res_module'):
                mid_layer = self.conv2d(inputs=self.mid_res_X, filters=32, kernel_size=3, strides=2, name='conv2d_0')
                mid_layer = self.conv2d(inputs=mid_layer, filters=32, kernel_size=3, strides=1, name='conv2d_1')
                mid_layer = self.batch_norm(inputs=mid_layer, name='conv2d_0_batch')

                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=32, strides=2, repeat=3, factor=6, name='bottleneck_3')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
                mid_layer = self.inverted_bottleneck(inputs=mid_layer, filters=96, strides=1, repeat=3, factor=6, name='bottleneck_5')
                mid_layer = self.conv2d(inputs=mid_layer, filters=320, name='conv2d_6')
                mid_layer = self.batch_norm(inputs=mid_layer, name='conv2d_6_batch')

                if self.is_logging:
                    self.summary_values.append(tf.summary.histogram('residual_network', mid_layer))

                # mid_layer = tf.layers.conv2d_transpose(inputs=mid_layer, filters=mid_layer.get_shape()[-1],
                #                                        kernel_size=(2, 2), strides=(1, 1), padding='VALID', name='resize')
                mid_layer = tf.image.resize_nearest_neighbor(images=mid_layer, size=(7, 15), name='resize')

            '''High Resolution Network'''
            with tf.variable_scope(name_or_scope='high_res_module'):
                high_layer = self.conv2d(inputs=self.high_res_X, filters=32, kernel_size=3, strides=2, name='conv2d_0')
                high_layer = self.conv2d(inputs=high_layer, filters=32, kernel_size=3, strides=1, name='conv2d_1')
                high_layer = self.conv2d(inputs=high_layer, filters=32, kernel_size=3, strides=1, name='conv2d_2')
                high_layer = self.batch_norm(inputs=high_layer, name='conv2d_0_batch')

                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=32, strides=2, repeat=3, factor=6, name='bottleneck_3')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
                high_layer = self.inverted_bottleneck(inputs=high_layer, filters=96, strides=1, repeat=3, factor=6, name='bottleneck_5')
                high_layer = self.conv2d(inputs=high_layer, filters=320, name='conv2d_6')
                high_layer = self.batch_norm(inputs=high_layer, name='conv2d_6_batch')

                if self.is_logging:
                    self.summary_values.append(tf.summary.histogram('residual_network', high_layer))

            with tf.variable_scope('multi_scale_module'):
                tot_layer = tf.concat([low_layer, mid_layer, high_layer], axis=-1, name='multiscale_concat')

                tot_layer = self.conv2d(inputs=tot_layer, filters=320, name='tot_bottleneck')
                tot_layer = self.batch_norm(inputs=tot_layer, name='tot_batch')

                a_icpt_layer = self.inception_resnet_A(inputs=tot_layer, filters=320, name='inception_module_A')
                icpt_layer_a = self.squeeze_excitation(inputs=a_icpt_layer, num_outputs=320, name='squeeze_excitation_A')
                icpt_layer_a = self.conv2d(inputs=icpt_layer_a, filters=100, name='conv2d_A')
                icpt_layer_a = tf.reduce_mean(input_tensor=icpt_layer_a, axis=[1, 2], keep_dims=True, name='global_pool_A')
                # icpt_layer_a = tf.squeeze(input=icpt_layer_a, axis=[1, 2], name='squeeze_A')

                b_icpt_layer = self.inception_resnet_A(inputs=a_icpt_layer, filters=320, name='inception_module_B')
                icpt_layer_b = self.squeeze_excitation(inputs=b_icpt_layer, num_outputs=320, name='squeeze_excitation_B')
                icpt_layer_b = self.conv2d(inputs=icpt_layer_b, filters=100, name='conv2d_B')
                icpt_layer_b = tf.reduce_mean(input_tensor=icpt_layer_b, axis=[1, 2], keep_dims=True, name='global_pool_B')
                # icpt_layer_b = tf.squeeze(input=icpt_layer_b, axis=[1, 2], name='squeeze_B')

                icpt_layer_c = tf.layers.max_pooling2d(inputs=b_icpt_layer, pool_size=2, strides=2, padding='same', name='max_pool_C')
                icpt_layer_c = self.inception_resnet_A(inputs=icpt_layer_c, filters=320, name='inception_module_C')
                icpt_layer_c = self.squeeze_excitation(inputs=icpt_layer_c, num_outputs=320, name='squeeze_excitation_C')
                icpt_layer_c = self.conv2d(inputs=icpt_layer_c, filters=100, name='conv2d_C')
                icpt_layer_c = tf.reduce_mean(input_tensor=icpt_layer_c, axis=[1, 2], keep_dims=True, name='global_pool_C')
                # icpt_layer_c = tf.squeeze(input=icpt_layer_c, axis=[1, 2], name='squeeze_C')

                last_layer = tf.concat([icpt_layer_a, icpt_layer_b, icpt_layer_c], axis=-1, name='inception_concat')

                self.cam_layer = last_layer

                if self.is_logging:
                    self.summary_values.append(tf.summary.histogram('output_network', tot_layer))

                last_layer = self.dropout(inputs=last_layer, rate=flags.FLAGS.dropout_rate, name='last_dropout')
                last_layer = self.conv2d(inputs=last_layer, filters=7, name='conv2d_output')
                self.logits = tf.squeeze(input=last_layer, axis=[1, 2], name='squeeze_output')

            with tf.variable_scope('output_module'):
                self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

                self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
                self.loss = tf.add_n([self.loss] +
                                     [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float32))

                if self.is_logging:
                    self.summary_values.append(tf.summary.scalar('loss', self.loss))
                    self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

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