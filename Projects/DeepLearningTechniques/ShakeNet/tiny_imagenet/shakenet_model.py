from Projects.Hongbog.EyeOclock.cifar10.shakenet_constants import *


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

        self.kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=flags.FLAGS.l2_scale)

        self.k = 2
        self.num_blocks = 5

        self.summary_values = []
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name_or_scope='input_scope'):
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
                self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
                self.is_train = tf.Variable(self.is_training, trainable=False, dtype=tf.bool, name='is_training')
                self.global_step = tf.Variable(0, trainable=False, name='global_step')

            with tf.variable_scope(name_or_scope='body_scope'):
                x = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=2, name='conv2d_0')
                x = self.batch_norm(inputs=x, name='conv2d_0_batch')

                # todo shake stage-1
                x = self.shake_stage(x=x, output_filters=32 * self.k, num_blocks=self.num_blocks, strides=1, name='shake_stage_1')

                if self.is_tb_logging:
                    self.summary_values.append(tf.summary.histogram('shake_stage-1', x))

                # todo shake stage-2
                x = self.shake_stage(x=x, output_filters=32 * 2 * self.k, num_blocks=self.num_blocks, strides=2, name='shake_stage_2')

                if self.is_tb_logging:
                    self.summary_values.append(tf.summary.histogram('shake_stage-2', x))

                # todo shake stage-3
                x = self.shake_stage(x=x, output_filters=32 * 4 * self.k, num_blocks=self.num_blocks, strides=2, name='shake_stage_3')

                if self.is_tb_logging:
                    self.summary_values.append(tf.summary.histogram('shake_stage-3', x))

                x = tf.nn.relu(x, name='relu_5')

                self.cam_layer = x

                x = tf.reduce_mean(x, axis=[1, 2], keepdims=False, name='global_avg_pool_5')
                self.logits = self.dense(inputs=x, units=flags.FLAGS.image_class, name='dense_5')

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

                # self.decay_lr = tf.train.exponential_decay(self.lr, self.global_step, 500, flags.FLAGS.learning_rate_decay, staircase=True)
                # self.decay_lr = tf.train.cosine_decay_restarts(self.lr, self.global_step, 390)
                self.decay_lr = tf.train.cosine_decay(self.lr, self.global_step, 781 * flags.FLAGS.epoch)

                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.decay_lr, momentum=0.9, use_nesterov=True)
                # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.decay_lr)
                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.decay_lr)
                # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.decay_lr)

                update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
                with tf.control_dependencies(update_opt):
                    self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

                if self.is_tb_logging:
                    self.summary_merged_values = tf.summary.merge(inputs=self.summary_values)

    def shake_stage(self, x, output_filters, num_blocks, strides, name):
        num_branches = 2

        def shake_skip_connection(x, output_filters, strides, name):
            with tf.variable_scope(name_or_scope=name):
                input_filters = int(x.get_shape()[-1])

                if input_filters == output_filters:
                    return x

                x = tf.nn.relu(x, name='relu')

                with tf.variable_scope(name_or_scope='skip_connection_0'):
                    sc1 = tf.layers.average_pooling2d(inputs=x, pool_size=1, strides=strides, padding='valid', name='avg_pool')
                    sc1 = self.conv2d(inputs=sc1, filters=int(output_filters / 2), kernel_size=1, strides=1, name='conv2d')

                with tf.variable_scope(name_or_scope='skip_connection_1'):
                    sc2 = tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]])[:, 1:, 1:, :]  # 각 차원 별 [앞, 뒤] padding
                    sc2 = tf.layers.average_pooling2d(inputs=sc2, pool_size=1, strides=strides, padding='valid', name='avg_pool')
                    sc2 = self.conv2d(inputs=sc2, filters=int(output_filters / 2), kernel_size=1, strides=1, name='conv2d')

                with tf.variable_scope('concat'):
                    x = tf.concat([sc1, sc2], axis=-1, name='concat')
                    x = self.batch_norm(inputs=x, act=None, name='batch_norm')

            return x

        def shake_branch(x, output_filters, strides, forward, backward, name):
            with tf.variable_scope(name_or_scope=name):
                x = tf.nn.relu(x, name='conv2d_0_relu')
                x = self.conv2d(inputs=x, filters=output_filters, kernel_size=3, strides=strides, name='conv2d_0')
                x = self.batch_norm(inputs=x, act=None, name='conv2d_0_batch')

                x = tf.nn.relu(x, name='conv2d_1_relu')
                x = self.conv2d(inputs=x, filters=output_filters, kernel_size=3, strides=1, name='conv2d_1')
                x = self.batch_norm(inputs=x, act=None, name='conv2d_1_batch')

                x = tf.cond(self.is_train, lambda: x * backward + tf.stop_gradient(x * forward - x * backward), lambda: x / num_branches)

            return x

        def shake_block(x, output_filters, strides, name):
            with tf.variable_scope(name_or_scope=name):
                rand_forward = [tf.random_uniform([flags.FLAGS.batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32) for _ in range(num_branches)]
                rand_backward = [tf.random_uniform([flags.FLAGS.batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32) for _ in range(num_branches)]

                tot_forward, tot_backward = tf.add_n(rand_forward), tf.add_n(rand_backward)
                norm_forward = [forward / tot_forward for forward in rand_forward]
                norm_backward = [backward / tot_backward for backward in rand_backward]

                pair_rand = zip(norm_forward, norm_backward)

                branches = []
                for branch_idx, (forward, backward) in enumerate(pair_rand):
                    b = shake_branch(x=x, output_filters=output_filters, strides=strides, forward=forward, backward=backward, name='shake_branch_' + str(branch_idx))
                    branches.append(b)

                x = shake_skip_connection(x, output_filters, strides, name)

            return x + tf.add_n(branches)

        with tf.variable_scope(name_or_scope=name):
            for block_idx in range(num_blocks):
                x = shake_block(x=x, output_filters=output_filters, strides=strides if block_idx == 0 else 1, name='shake_block_' + str(block_idx))

        return x

    def batch_norm(self, inputs, act=tf.nn.relu6, name='batch_norm_layer'):
        '''
            Batch Normalization
             - scale=True, scale factor(gamma) 를 사용
             - center=True, shift factor(beta) 를 사용
        '''
        with tf.variable_scope(name_or_scope=name):
            return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, center=True, scale=True, fused=True,
                                                updates_collections=tf.GraphKeys.UPDATE_OPS, activation_fn=act,
                                                is_training=self.is_train, scope='batch_norm')

    def conv2d(self, inputs, filters, kernel_size=1, strides=1, padding='same', act=tf.identity, name='conv2d'):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=padding, activation=act, use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                name=name)

    def dense(self, inputs, units, act=tf.identity, name='dense'):
        return tf.layers.dense(inputs=inputs, units=units, activation=act, use_bias=False,
                               kernel_initializer=self.kernel_initializer,
                               kernel_regularizer=self.kernel_regularizer,
                               name=name)

    def dropout(self, inputs, rate, name):
        with tf.variable_scope(name_or_scope=name):
            return tf.layers.dropout(inputs=inputs, rate=rate, training=self.is_train, name='dropout')

    def train(self, x, y):
        if self.is_tb_logging:
            return self.sess.run([self.accuracy, self.loss, self.summary_merged_values, self.train_op], feed_dict={self.x: x, self.y: y})
        else:
            return self.sess.run([self.accuracy, self.loss, self.train_op], feed_dict={self.x: x, self.y: y})

    def test(self, x, y):
        return self.sess.run([self.accuracy, self.loss, self.prob], feed_dict={self.x: x, self.y: y})