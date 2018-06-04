import tensorflow as tf
import tensorflow.contrib.slim as slim

class Model:
    def __init__(self, sess, scope):
        self.sess = sess
        self.scope = scope
        self.N = 12
        self.growthRate = 10
        self.compression_factor = 0.5
        self.hidden_num = 16
        self._build_graph()

    def _build_graph(self):
        def _batch_norm(inputs, act=tf.nn.relu6, scope='batch_norm'):
            with slim.arg_scope([slim.batch_norm], decay=0.999, epsilon=0.001, reuse=False, zero_debias_moving_mean=True, scope=scope):
                return slim.batch_norm(inputs=inputs, activation_fn=act, is_training=self.training)

        def _dense_block(inputs, scope):
            with tf.variable_scope(scope):
                '''bottleneck layer (DenseNet-B)'''
                layer = _batch_norm(inputs=inputs, scope='bottleneck_batch_norm')
                layer = slim.dropout(inputs=layer, keep_prob=self.dropout_rate, is_training=self.training, scope='bottleneck_dropout')
                layer = slim.conv2d(inputs=layer, num_outputs=4 * self.growthRate, kernel_size=1, stride=1, padding='SAME', scope='bottleneck_conv')

                '''basic dense layer'''
                layer = _batch_norm(inputs=layer, scope='basic_batch_norm')
                layer = slim.dropout(inputs=layer, keep_prob=self.dropout_rate, is_training=self.training, scope='basic_dropout_a')
                layer = slim.separable_convolution2d(inputs=layer, num_outputs=self.growthRate, kernel_size=3, depth_multiplier=1, stride=1,
                                                     padding='SAME', scope='basic_dwconv')
                layer = slim.dropout(inputs=layer, keep_prob=self.dropout_rate, is_training=self.training, scope='basic_dropout_b')
                layer = tf.concat([inputs, layer], axis=-1, name='dense_concat')
            return layer

        def _transition_layer(inputs, scope):
            with tf.variable_scope(scope):
                '''compression transition layer (DenseNet-C)'''
                layer = _batch_norm(inputs=inputs, scope='transition_batch_norm')
                layer = slim.dropout(inputs=layer, keep_prob=self.dropout_rate, is_training=self.training, scope='transition_dropout')
                layer = slim.conv2d(inputs=layer, num_outputs=4 * self.growthRate, kernel_size=1, stride=1, padding='SAME', scope='bottleneck_conv')
                layer = slim.max_pool2d(inputs=layer, kernel_size=2, stride=1, padding='SAME', scope='transition_pooling')
            return layer

        def _residual_block(inputs, num_outputs, kernel_size, stride, padding='SAME', scope='residual_block'):
            with tf.variable_scope(scope):
                layer = _batch_norm(inputs=inputs, scope='batch_norm_1')
                layer = slim.conv2d(inputs=layer, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding, scope='residual_conv_1')
                layer = _batch_norm(inputs=layer, scope='batch_norm_2')
                layer = slim.conv2d(inputs=layer, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding, scope='residual_conv_2')
                layer = tf.add(inputs, layer, name='residual_add')
            return layer

        def _inception_resnet_a(inputs, scope):
            with tf.variable_scope(scope):
                sub_layer_a = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a01')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=48, kernel_size=3, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a02')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a03')

                sub_layer_b = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_b01')
                sub_layer_b = slim.conv2d(inputs=sub_layer_b, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_b02')

                sub_layer_c = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_c01')

                layer = tf.concat([sub_layer_a, sub_layer_b, sub_layer_c], axis=-1, name='concat_layer')
                layer = slim.conv2d(inputs=layer, num_outputs=384, kernel_size=1, stride=1, activation_fn=tf.identity, scope='output_layer')
                layer = tf.add(inputs, layer, name='add_layer')
            return layer

        def _inception_resnet_b(inputs, scope):
            with tf.variable_scope(scope):
                sub_layer_a = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a01')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=48, kernel_size=(1, 5), stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a02')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=64, kernel_size=(5, 1), stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a03')

                sub_layer_b = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_b01')

                layer = tf.concat([sub_layer_a, sub_layer_b], axis=-1, name='concat_layer')
                layer = slim.conv2d(inputs=layer, num_outputs=128, kernel_size=1, stride=1, activation_fn=tf.identity, scope='output_layer')
                layer = tf.add(inputs, layer, name='add_layer')
            return layer

        def _inception_resnet_c(inputs, scope):
            with tf.variable_scope(scope):
                sub_layer_a = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a01')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=96, kernel_size=(1, 3), stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a02')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=128, kernel_size=(3, 1), stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a03')

                sub_layer_b = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_b01')

                layer = tf.concat([sub_layer_a, sub_layer_b], axis=-1, name='concat_layer')
                layer = slim.conv2d(inputs=layer, num_outputs=512, kernel_size=1, stride=1, activation_fn=tf.identity, scope='output_layer')
                layer = tf.add(inputs, layer, name='add_layer')
            return layer

        def _reduction_a(inputs, scope):
            with tf.variable_scope(scope):
                sub_layer_a = slim.conv2d(inputs=inputs, num_outputs=96, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a01')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=96, kernel_size=3, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a02')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu6, scope='sub_layer_a03')

                sub_layer_b = slim.conv2d(inputs=inputs, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu6, scope='sub_layer_b01')

                sub_layer_c = slim.max_pool2d(inputs=inputs, kernel_size=3, stride=2, padding='SAME', scope='sub_layer_c01')

                layer = tf.concat([sub_layer_a, sub_layer_b, sub_layer_c], axis=-1, name='concat_layer')
            return layer


        def _reduction_b(inputs, scope):
            with tf.variable_scope(scope):
                sub_layer_a = slim.conv2d(inputs=inputs, num_outputs=48, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a01')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=72, kernel_size=3, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a02')
                sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu6, scope='sub_layer_a03')

                sub_layer_b = slim.conv2d(inputs=inputs, num_outputs=48, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_b01')
                sub_layer_b = slim.conv2d(inputs=sub_layer_b, num_outputs=96, kernel_size=3, stride=2, activation_fn=tf.nn.relu6, scope='sub_layer_b02')

                sub_layer_c = slim.conv2d(inputs=inputs, num_outputs=48, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_c01')
                sub_layer_c = slim.conv2d(inputs=sub_layer_c, num_outputs=160, kernel_size=3, stride=2, activation_fn=tf.nn.relu6, scope='sub_layer_c02')

                sub_layer_d = slim.max_pool2d(inputs=inputs, kernel_size=3, stride=2, padding='SAME', scope='sub_layer_d01')

                layer = tf.concat([sub_layer_a, sub_layer_b, sub_layer_c, sub_layer_d], axis=-1, name='concat_layer')
            return layer

        def _network():
            with tf.variable_scope(self.scope):
                with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                                    weights_initializer=self.initializer, weights_regularizer=self.regularizer):
                    with tf.variable_scope('input_network'):
                        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 60, 160, 1], name='x_data')
                        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_data')

                        self.training = tf.placeholder(dtype=tf.bool, shape=None, name='training')
                        self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=None, name='dropout_rate')
                        self.initializer = tf.truncated_normal_initializer(stddev=0.01)
                        self.regularizer = slim.l2_regularizer(0.0005)

                    '''stem-layer part 1'''
                    with tf.variable_scope('stem_layer_a'):
                        layer = slim.conv2d(inputs=self.x, num_outputs=16, kernel_size=3, stride=2, padding='VALID', activation_fn=tf.nn.relu6, scope='conv_01')
                        layer = slim.conv2d(inputs=layer, num_outputs=16, kernel_size=3, stride=1, activation_fn=tf.nn.relu6, scope='conv_02')
                        layer = slim.conv2d(inputs=layer, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu6, scope='conv_03')

                        sub_layer_a = slim.conv2d(inputs=layer, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu6, scope='sub_layer_a01')
                        sub_layer_b = slim.max_pool2d(inputs=layer, kernel_size=3, stride=2, padding='SAME', scope='sub_layer_b01')

                        layer = tf.concat([sub_layer_a, sub_layer_b], axis=-1, name='concat_layer')

                    '''stem-layer part 2'''
                    with tf.variable_scope('stem_layer_b'):
                        sub_layer_a = slim.conv2d(inputs=layer, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a02')
                        sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=32, kernel_size=(1, 5), stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a03')
                        sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=32, kernel_size=(5, 1), stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_a04')
                        sub_layer_a = slim.conv2d(inputs=sub_layer_a, num_outputs=64, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu6, scope='sub_layer_a05')

                        sub_layer_b = slim.conv2d(inputs=layer, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu6, scope='sub_layer_b02')
                        sub_layer_b = slim.conv2d(inputs=sub_layer_b, num_outputs=64, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu6, scope='sub_layer_b03')

                        layer = tf.concat([sub_layer_a, sub_layer_b], axis=-1, name='concat_layer')

                    '''inception-resnet-B'''
                    with tf.variable_scope('inception_resnet_B'):
                        for i in range(6):
                            layer = _inception_resnet_b(inputs=layer, scope='inception_resnet_B_{}'.format(i))

                    layer = _reduction_b(inputs=layer, scope='reduction_B')

                    '''inception-resnet-C'''
                    with tf.variable_scope('inception_resnet_C'):
                        for i in range(3):
                            layer = _inception_resnet_c(inputs=layer, scope='inception_resnet_C_{}'.format(i))

                    '''dense-block'''
                    with tf.variable_scope('dense_block'):
                        for i in range(self.N):
                            layer = _dense_block(inputs=layer, scope='dense_block_{}'.format(i))
                        layer = _transition_layer(inputs=layer, scope='transition_layer')

                    self.cam_layer = layer

                    with tf.variable_scope('output_network'):
                        layer = slim.dropout(inputs=layer, keep_prob=self.dropout_rate, is_training=self.training, scope='dropout')
                        layer = slim.conv2d(inputs=layer, num_outputs=7, kernel_size=1, stride=1, scope='logit')
                        layer = slim.avg_pool2d(inputs=layer, kernel_size=(4, 8), stride=1, padding='VALID', scope='global_avg_pool')

            self.variables = [var for var in tf.trainable_variables() if self.name in var.name]

            return layer

        self.logits = _network()
        self.logits = tf.squeeze(self.logits, [1, 2], name=self.scope+'/logits/squeezed')
        self.prob = slim.softmax(logits=self.logits, scope=self.scope+'/softmax')
        self.loss = slim.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.y, scope=self.scope+'/ce_loss') \
                    + tf.add_n(slim.losses.get_regularization_losses())
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float32), name=self.scope+'/acc')

        update_ops = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.scope in var.name]
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name=self.scope+'/opt').minimize(self.loss, var_list=self.variables)

    def predict(self, x, training, dropout_rate=1.0):
        return self.sess.run(self.prob,
                             feed_dict={self.x: x, self.training: training, self.dropout_rate: dropout_rate})

    def train(self, x, y, training, dropout_rate=0.6):
        return self.sess.run([self.accuracy, self.loss, self.optimizer],
                             feed_dict={self.x: x, self.y: y, self.training: training, self.dropout_rate: dropout_rate})

    def validation(self, x, y, training, dropout_rate=1.0):
        return self.sess.run([self.accuracy, self.loss, self.prob],
                             feed_dict={self.x: x, self.y: y, self.training: training, self.dropout_rate: dropout_rate})