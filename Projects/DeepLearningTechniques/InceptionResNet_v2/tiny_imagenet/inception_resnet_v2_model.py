from Projects.DeepLearningTechniques.InceptionResNet_v2.tiny_imagenet.constants import *

class Model:
    def __init__(self, sess, is_training, is_tb_logging, name):
        self.sess = sess
        self.is_training = is_training
        self.is_tb_logging = is_tb_logging
        self.name = name

        self.width = flags.FLAGS.image_width
        self.height = flags.FLAGS.image_height
        self.channel = flags.FLAGS.image_channel
        self.lr = flags.FLAGS.learning_rate
        self.dr = flags.FLAGS.learning_rate_decay
        self.growthRate = 12

        self.weights_initializers = tf.contrib.layers.xavier_initializer()
        self.weights_regularizers = tf.contrib.layers.l2_regularizer(scale=flags.FLAGS.l2_scale)

        self.summary_values = []
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name_or_scope='input_scope'):
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
                self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
                self.global_step = tf.Variable(0, trainable=False)

            #todo Stem Layer, 28x28, 96
            with tf.variable_scope(name_or_scope='body_scope'):
                with tf.variable_scope(name_or_scope='stem_layer'):
                    layer = self.conv2d_bn(inputs=self.x, filters=8, kernel_size=3, strides=1, padding='valid', act=tf.nn.elu, name='conv_01')
                    layer = self.conv2d_bn(inputs=layer, filters=8, kernel_size=3, strides=1, padding='valid', act=tf.nn.elu, name='conv_02')
                    layer = self.conv2d_bn(inputs=layer, filters=16, kernel_size=3, strides=1, padding='same', act=tf.nn.elu, name='conv_03')
                    layer_a = self.conv2d_bn(inputs=layer, filters=16, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_04_a01')
                    layer_b = self.conv2d_bn(inputs=layer, filters=32, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_04_b01')
                    layer = tf.concat([layer_a, layer_b], axis=-1, name='concat_01')
                    layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='dropout_01')

                    layer_a = self.conv2d_bn(inputs=layer, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_05_a01')
                    layer_a = self.conv2d_bn(inputs=layer_a, filters=48, kernel_size=3, strides=1, padding='valid', act=tf.nn.elu, name='conv_05_a02')
                    layer_b = self.conv2d_bn(inputs=layer, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_05_b01')
                    layer_b = self.conv2d_bn(inputs=layer_b, filters=32, kernel_size=(5, 1), strides=1, padding='same', act=tf.nn.elu, name='conv_05_b02')
                    layer_b = self.conv2d_bn(inputs=layer_b, filters=32, kernel_size=(1, 5), strides=1, padding='same', act=tf.nn.elu, name='conv_05_b03')
                    layer_b = self.conv2d_bn(inputs=layer_b, filters=48, kernel_size=3, strides=1, padding='valid', act=tf.nn.elu, name='conv_05_b04')
                    stem_layer = tf.concat([layer_a, layer_b], axis=-1, name='concat_02')
                    stem_layer = self.dropout(inputs=stem_layer, rate=flags.FLAGS.dropout_rate, name='dropout_02')

                #todo Inception-resnet-A, 28x28, 96
                for idx in range(5):
                    if idx == 0:
                        layer = self.inception_resnet_A(inputs=stem_layer, name='inception_resnet_A_' + str(idx))
                    else:
                        layer = self.inception_resnet_A(inputs=layer, name='inception_resnet_A_' + str(idx))

                #todo Reduction-A, 14x14, 160
                red_layer_A = self.reduction_A(inputs=layer, name='reduction_A')
                red_layer_A = self.dropout(inputs=red_layer_A, rate=flags.FLAGS.dropout_rate, name='dropout_red_A')

                #todo Inception-resnet-B, 14x14, 160
                for idx in range(10):
                    if idx == 0:
                        layer = self.inception_resnet_B(inputs=red_layer_A, name='inception_resnet_B_' + str(idx))
                    else:
                        layer = self.inception_resnet_B(inputs=layer, name='inception_resnet_B_' + str(idx))

                #todo Reduction-B, 7x7, 240
                red_layer_B = self.reduction_B(inputs=layer, name='reduction_B')
                red_layer_B = self.dropout(inputs=red_layer_B, rate=flags.FLAGS.dropout_rate, name='dropout_red_B')

                #todo Inception-resnet-C, 7x7, 240
                for idx in range(5):
                    if idx == 0:
                        layer = self.inception_resnet_C(inputs=red_layer_B, name='inception_resnet_C_' + str(idx))
                    else:
                        layer = self.inception_resnet_C(inputs=layer, name='inception_resnet_C_' + str(idx))

                #todo Reduction-C, 4x4, 400
                red_layer_C = self.reduction_C(inputs=layer, name='reduction_C')
                red_layer_C = self.dropout(inputs=red_layer_C, rate=flags.FLAGS.dropout_rate, name='dropout_red_C')

                #todo Reduction-C, Upsampling
                red_layer_C = tf.image.resize_nearest_neighbor(red_layer_C, red_layer_B.get_shape()[1:3], name='red_c_resize')

                #todo Reduction-B, DenseBlock & Upsampling
                red_layer_B = self.dense_block(inputs=red_layer_B, repeat=4, name='red_b_dense_block_01')
                red_layer_B = self.dense_block(inputs=red_layer_B, repeat=4, name='red_b_dense_block_02')
                red_layer_B = tf.concat([red_layer_B, red_layer_C], axis=-1, name='red_b_concat')
                red_layer_B = self.dense_block(inputs=red_layer_B, repeat=4, name='red_b_dense_block_03')
                red_layer_B = tf.image.resize_nearest_neighbor(red_layer_B, red_layer_A.get_shape()[1:3], name='red_b_resize')

                #todo Reduction-A, DenseBlock & Upsamling
                red_layer_A = self.dense_block(inputs=red_layer_A, repeat=3, name='red_a_dense_block_01')
                red_layer_A = self.dense_block(inputs=red_layer_A, repeat=3, name='red_a_dense_block_02')
                red_layer_A = tf.concat([red_layer_A, red_layer_B], axis=-1, name='red_a_concat')
                red_layer_A = self.dense_block(inputs=red_layer_A, repeat=3, name='red_a_dense_block_03')
                red_layer_A = tf.image.resize_nearest_neighbor(red_layer_A, stem_layer.get_shape()[1:3], name='red_a_resize')

                #todo Stem, DenseBlock
                stem_layer = self.dense_block(inputs=stem_layer, repeat=2, name='stem_dense_block_01')
                stem_layer = self.dense_block(inputs=stem_layer, repeat=2, name='stem_dense_block_02')
                stem_layer = tf.concat([stem_layer, red_layer_A], axis=-1, name='stem_concat')
                stem_layer = self.dense_block(inputs=stem_layer, repeat=2, name='stem_dense_block_03')

                #todo Global Average Pooling
                self.cam_layer = stem_layer
                layer = self.dropout(inputs=stem_layer, rate=flags.FLAGS.dropout_rate, name='last_dropout')
                layer = tf.layers.average_pooling2d(inputs=layer, pool_size=28, strides=1, name='last_avg_pool')
                layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='last_conv')
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

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.decay_lr)
                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.decay_lr)
                # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.decay_lr)

                update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
                with tf.control_dependencies(update_opt):
                    self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

                if self.is_tb_logging:
                    self.summary_merged_values = tf.summary.merge(inputs=self.summary_values)


    def inception_resnet_A(self, inputs, name):
        with tf.variable_scope(name_or_scope=name):
            layer_a = self.conv2d_bn(inputs=inputs, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_a01')
            layer_b = self.conv2d_bn(inputs=inputs, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_b01')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=16, kernel_size=3, strides=1, padding='same', act=tf.nn.elu, name='conv_01_b02')
            layer_c = self.conv2d_bn(inputs=inputs, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_c01')
            layer_c = self.conv2d_bn(inputs=layer_c, filters=32, kernel_size=3, strides=1, padding='same', act=tf.nn.elu, name='conv_01_c02')
            layer_c = self.conv2d_bn(inputs=layer_c, filters=40, kernel_size=3, strides=1, padding='same', act=tf.nn.elu, name='conv_01_c03')
            layer_d = self.conv2d(inputs=tf.concat([layer_a, layer_b, layer_c], axis=-1),
                                  filters=96, kernel_size=1, strides=1, padding='same', name='conv_02_d01')
            layer = tf.add(layer_d, inputs, name='shortcut_conn')
            layer = tf.nn.elu(layer, name='elu')

            return layer

    def reduction_A(self, inputs, name):
        with tf.variable_scope(name_or_scope=name):
            layer_a = self.conv2d_bn(inputs=inputs, filters=96, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_01_a01')
            layer_b = self.conv2d_bn(inputs=inputs, filters=16, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_01_b01')
            layer_c = self.conv2d_bn(inputs=inputs, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_c01')
            layer_c = self.conv2d_bn(inputs=layer_c, filters=32, kernel_size=3, strides=1, padding='same', act=tf.nn.elu, name='conv_01_c02')
            layer_c = self.conv2d_bn(inputs=layer_c, filters=48, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_01_c03')

            return tf.concat([layer_a, layer_b, layer_c], axis=-1, name='concat')

    def inception_resnet_B(self, inputs, name):
        with tf.variable_scope(name_or_scope=name):
            layer_a = self.conv2d_bn(inputs=inputs, filters=32, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_a01')
            layer_b = self.conv2d_bn(inputs=inputs, filters=32, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_b01')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=48, kernel_size=(1, 7), strides=1, padding='same', act=tf.nn.elu, name='conv_01_b02')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=64, kernel_size=(7, 1), strides=1, padding='same', act=tf.nn.elu, name='conv_01_b03')
            layer_c = self.conv2d_bn(inputs=tf.concat([layer_a, layer_b], axis=-1),
                                     filters=160, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_02_c01')
            layer = tf.add(layer_c, inputs, name='shortcut_conn')
            layer = tf.nn.elu(layer, name='elu')

            return layer

    def reduction_B(self, inputs, name):
        with tf.variable_scope(name_or_scope=name):
            layer_a = self.conv2d_bn(inputs=inputs, filters=96, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_01_a01')
            layer_b = self.conv2d_bn(inputs=inputs, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_b01')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=32, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_01_b02')
            layer_c = self.conv2d_bn(inputs=inputs, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_c01')
            layer_c = self.conv2d_bn(inputs=layer_c, filters=32, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_01_c02')
            layer_d = self.conv2d_bn(inputs=inputs, filters=32, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_d01')
            layer_d = self.conv2d_bn(inputs=layer_d, filters=48, kernel_size=3, strides=1, padding='same', act=tf.nn.elu, name='conv_01_d02')
            layer_d = self.conv2d_bn(inputs=layer_d, filters=80, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_01_d03')

            return tf.concat([layer_a, layer_b, layer_c, layer_d], axis=-1, name='concat')

    def inception_resnet_C(self, inputs, name):
        with tf.variable_scope(name_or_scope=name):
            layer_a = self.conv2d_bn(inputs=inputs, filters=96, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_a01')
            layer_b = self.conv2d_bn(inputs=inputs, filters=64, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_01_b01')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=80, kernel_size=(1, 3), strides=1, padding='same', act=tf.nn.elu, name='conv_01_b02')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=96, kernel_size=(3, 1), strides=1, padding='same', act=tf.nn.elu, name='conv_01_b03')
            layer_c = self.conv2d_bn(inputs=tf.concat([layer_a, layer_b], axis=-1),
                                     filters=240, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_02_c01')
            layer = tf.add(layer_c, inputs, name='shortcut_conn')
            layer = tf.nn.elu(layer, name='elu')

            return layer

    def reduction_C(self, inputs, name):
        with tf.variable_scope(name_or_scope=name):
            layer_a = tf.layers.average_pooling2d(inputs=inputs, pool_size=3, strides=2, padding='same', name='avg_pool')
            layer_b = self.conv2d_bn(inputs=inputs, filters=80, kernel_size=3, strides=2, padding='same', name='conv_01_b01')
            layer_c = self.conv2d_bn(inputs=inputs, filters=48, kernel_size=1, strides=1, padding='same', name='conv_01_c01')
            layer_c = self.conv2d_bn(inputs=layer_c, filters=80, kernel_size=3, strides=2, padding='same', name='conv_01_c02')

            return tf.concat([layer_a, layer_b, layer_c], axis=-1, name='concat')

    def dense_block(self, inputs, repeat, name):
        def sub_dense_block(inputs, name):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                layer = self.batch_norm(inputs=inputs, act=None, name='bottleneck_batch_norm')
                layer = tf.nn.elu(layer, 'bottleneck_elu')
                layer = self.conv2d(inputs=layer, filters=4 * self.growthRate, kernel_size=1, strides=1, padding='same', name='bottleneck_conv')

                '''basic dense layer'''
                layer = self.batch_norm(inputs=layer, act=None, name='basic_batch_norm')
                layer = tf.nn.elu(layer, 'basic_elu')
                layer = self.conv2d(inputs=layer, filters=self.growthRate, kernel_size=3, strides=1, padding='same', name='basic_conv')

                layer = tf.concat([inputs, layer], axis=-1)

                return layer

        for idx in range(repeat):
            if idx == 0:
                layer = sub_dense_block(inputs=inputs, name=name + '_' + str(idx))
            else:
                layer = sub_dense_block(inputs=layer, name=name + '_' + str(idx))

        return layer

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

    def conv2d_bn(self, inputs, filters, kernel_size=1, strides=1, padding='same', act=tf.identity, name='conv2d_layer'):
        with tf.variable_scope(name_or_scope=name):
            layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                     padding=padding, activation=act,
                                     kernel_initializer=self.weights_initializers,
                                     bias_initializer=self.weights_initializers,
                                     kernel_regularizer=self.weights_regularizers,
                                     bias_regularizer=self.weights_regularizers,
                                     name='conv')
            layer = tf.contrib.layers.batch_norm(inputs=layer, decay=0.9, center=True, scale=True, fused=True,
                                                updates_collections=tf.GraphKeys.UPDATE_OPS, activation_fn=None,
                                                is_training=self.is_training, scope='batch_norm')

            return layer

    def conv2d(self, inputs, filters, kernel_size=1, strides=1, padding='same', act=tf.identity, name='conv2d_layer'):
        with tf.variable_scope(name_or_scope=name):
            return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                    padding=padding, activation=act,
                                    kernel_initializer=self.weights_initializers,
                                    bias_initializer=self.weights_initializers,
                                    kernel_regularizer=self.weights_regularizers,
                                    bias_regularizer=self.weights_regularizers,
                                    name='conv')

    def dropout(self, inputs, rate, name):
        with tf.variable_scope(name_or_scope=name):
            return tf.layers.dropout(inputs=inputs, rate=rate, training=self.is_training, name='dropout')

    def train(self, x, y):
        if self.is_tb_logging:
            return self.sess.run([self.accuracy, self.loss, self.summary_merged_values, self.train_op], feed_dict={self.x: x, self.y: y})
        else:
            return self.sess.run([self.accuracy, self.loss, self.train_op], feed_dict={self.x: x, self.y: y})

    def validation(self, x, y):
        return self.sess.run([self.accuracy, self.loss, self.prob], feed_dict={self.x: x, self.y: y})

    def test(self, x):
        return self.sess.run(self.prob, feed_dict={self.x: x})