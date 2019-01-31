from Projects.DeepLearningTechniques.NASNet.cifar10.nasnet_constants import *
import tensorflow.contrib.slim as slim

class Model:

    def __init__(self, sess, is_tb_logging, name):
        self.sess = sess
        self.width = flags.FLAGS.image_width
        self.height = flags.FLAGS.image_height
        self.channel = flags.FLAGS.image_channel
        self.lr = flags.FLAGS.learning_rate
        self.is_tb_logging = is_tb_logging
        self.name = name

        self.kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_OUT')
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=flags.FLAGS.l2_scale)

        self.normal_cell_num = 6
        self.num_conv_filters = 32
        self.filter_scaling_rate = 2

        self.summary_values = []
        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name_or_scope='input_scope'):
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
                self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
                self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
                self.global_step = tf.Variable(0, trainable=False, name='global_step')

            with tf.variable_scope(name_or_scope='body_scope'):
                x = self.conv2d(inputs=self.x, filters=self.num_conv_filters, kernel_size=3, strides=1, name='stem_conv2d')
                x = self.batch_norm(inputs=x, act=None, name='stem_batch_norm')

                prev_x = None

                #todo normal cell(1)
                for idx in range(self.normal_cell_num):
                    x, prev_x = self.normal_cell(x=x, prev_x=prev_x, num_filters=self.num_conv_filters, name='normal_cell_1_%s' % str(idx))

                #todo reduction cell(1)
                x, prev_x = self.reduction_cell(x=x, prev_x=prev_x, num_filters=self.num_conv_filters * self.filter_scaling_rate, name='reduction_cell_1')

                #todo normal cell(2)
                for idx in range(self.normal_cell_num):
                    x, prev_x = self.normal_cell(x=x, prev_x=prev_x, num_filters=self.num_conv_filters * self.filter_scaling_rate, name='normal_cell_2_%s' % str(idx))

                #todo reduction cell(2)
                x, prev_x = self.reduction_cell(x=x, prev_x=prev_x, num_filters=self.num_conv_filters * self.filter_scaling_rate ** 2, name='reduction_cell_2')

                #todo auxiliary branch
                self.aux_logits = self.auxiliary_branch(x=x, name='auxiliary_branch')

                #todo normal cell(3)
                for idx in range(self.normal_cell_num):
                    x, prev_x = self.normal_cell(x=x, prev_x=prev_x, num_filters=self.num_conv_filters * self.filter_scaling_rate ** 2, name='normal_cell_3_%s' % str(idx))

                self.cam_layer = x

                x = tf.nn.relu(x)
                x = tf.reduce_mean(x, axis=[1, 2], keepdims=False, name='global_avg_pool')
                self.logits = self.dense(inputs=x, units=flags.FLAGS.image_class, name='main_logits')

            with tf.variable_scope(name_or_scope='output_scope'):
                self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

                self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

                self.main_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='main_loss'))
                self.main_loss = tf.add_n([self.main_loss] +
                                          [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if 'auxiliary_branch' not in var.name], name='main_tot_loss')
                self.aux_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.aux_logits, labels=self.y, name='aux_loss'))
                self.aux_loss = tf.add_n([self.aux_loss] +
                                          [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if 'auxiliary_branch' in var.name], name='aux_tot_loss')
                self.loss = self.main_loss + 0.4 * self.aux_loss

                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, -1), self.y), dtype=tf.float32))

                if self.is_tb_logging:
                    self.summary_values.append(tf.summary.scalar('loss', self.loss))
                    self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

                #todo Default Learning rate: 1e-3
                threshold = 5.0

                self.decay_lr = tf.train.cosine_decay(self.lr, self.global_step, flags.FLAGS.step_per_epoch * flags.FLAGS.epoch)

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.decay_lr, momentum=0.9, epsilon=1.0)
                # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.decay_lr, momentum=0.9, use_nesterov=True)
                grads_and_vars = self.optimizer.compute_gradients(self.loss)
                cliped_grad = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]

                update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
                with tf.control_dependencies(update_opt):
                    self.train_op = self.optimizer.apply_gradients(grads_and_vars=cliped_grad, global_step=self.global_step)
                    # self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

                if self.is_tb_logging:
                    self.summary_merged_values = tf.summary.merge(inputs=self.summary_values)

    def auxiliary_branch(self, x, name='auxiliary_branch'):
        with tf.variable_scope(name_or_scope=name):
            aux_x = tf.nn.relu(x, name='aux_relu_1')
            aux_x = tf.nn.avg_pool(value=aux_x, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='VALID', name='aux_pool_1')
            aux_x = self.conv2d(inputs=aux_x, filters=128, name='aux_conv2d_1')
            aux_x = self.batch_norm(inputs=aux_x, act=None, name='aux_batch_norm_1')
            aux_x = tf.nn.relu(aux_x, name='aux_relu_2')
            aux_x = self.conv2d(inputs=aux_x, filters=768, kernel_size=(aux_x.get_shape()[1], aux_x.get_shape()[2]), padding='valid', name='aux_conv2d_2')
            aux_x = self.batch_norm(inputs=aux_x, act=None, name='aux_batch_norm_2')

            aux_x = tf.nn.relu(aux_x, name='aux_relu_3')
            aux_x = tf.squeeze(aux_x, axis=[1, 2], name='squeeze')
            aux_x = self.dense(inputs=aux_x, units=flags.FLAGS.image_class, name='aux_logits')

        return aux_x

    def adjust_block(self, x, prev_x, num_filters, name='adjust_block'):
        def skip_connection(x, num_filters, strides, name='adjust_reduction_block'):
            with tf.variable_scope(name_or_scope=name):
                x = tf.nn.relu(features=x)

                path1 = tf.nn.avg_pool(value=x, ksize=[1, 1, 1, 1], strides=[1, strides, strides, 1], padding='VALID', name='path1_avg_pool')
                path1 = self.conv2d(inputs=path1, filters=num_filters // 2, name='path1_conv2d')

                path2 = tf.pad(tensor=x, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])[:, 1:, 1:, :]
                path2 = tf.nn.avg_pool(value=path2, ksize=[1, 1, 1, 1], strides=[1, strides, strides, 1], padding='VALID', name='path2_avg_pool')
                path2 = self.conv2d(inputs=path2, filters=num_filters // 2, name='path2_conv2d')

                output = tf.concat(values=[path1, path2], axis=-1)
                output = self.batch_norm(inputs=output, act=None, name='batch_norm')

            return output

        with tf.variable_scope(name_or_scope=name):
            if prev_x is None:
                return x

            curr_num_shape = int(x.shape[2])
            prev_num_shape = int(prev_x.shape[2])
            prev_num_filters = int(prev_x.shape[-1])

            if curr_num_shape != prev_num_shape:  # 현재 normal cell 입력 shape 과 이전 cell 의 shape 이 같지 않을 경우
                prev_x = skip_connection(x=prev_x, num_filters=num_filters, strides=2, name='adjust_reduction_block')
            elif prev_num_filters != num_filters:
                with tf.variable_scope(name_or_scope='adjust_projection_block'):
                    prev_x = tf.nn.relu(features=prev_x)
                    prev_x = self.conv2d(inputs=prev_x, filters=num_filters, name='conv2d')
                    prev_x = self.batch_norm(inputs=prev_x, act=None, name='batch_norm')

        return prev_x

    def normal_cell(self, x, prev_x, num_filters, name='normal_cell'):
        with tf.variable_scope(name_or_scope=name):
            prev_x = self.adjust_block(x=x, prev_x=prev_x, num_filters=num_filters, name='adjust_block')

            h = tf.nn.relu(features=x)
            h = self.conv2d(inputs=h, filters=num_filters, name='normal_conv')
            h = self.batch_norm(inputs=h, act=None, name='normal_batch_norm')

            with tf.variable_scope(name_or_scope='block_1'):
                x1 = self.separable_conv2d_block(inputs=h, filters=num_filters, name='sep_conv2d')
                x1 = tf.add(x1, h, name='normal_add')

            with tf.variable_scope(name_or_scope='block_2'):
                x2_1 = self.separable_conv2d_block(inputs=prev_x, filters=num_filters, name='sep_conv2d_1')
                x2_2 = self.separable_conv2d_block(inputs=h, kernel_size=5, filters=num_filters, name='sep_conv2d_2')
                x2 = tf.add(x2_1, x2_2, name='normal_add')

            with tf.variable_scope(name_or_scope='block_3'):
                x3 = tf.nn.avg_pool(value=h, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='avg_pool')
                x3 = tf.add(x3, prev_x, name='normal_add')

            with tf.variable_scope(name_or_scope='block_4'):
                x4_1 = tf.nn.avg_pool(value=prev_x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='avg_pool_1')
                x4_2 = tf.nn.avg_pool(value=prev_x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='avg_pool_2')
                x4 = tf.add(x4_1, x4_2, name='normal_add')

            with tf.variable_scope(name_or_scope='block_5'):
                x5_1 = self.separable_conv2d_block(inputs=prev_x, kernel_size=5, filters=num_filters, name='sep_conv2d_1')
                x5_2 = self.separable_conv2d_block(inputs=prev_x, filters=num_filters, name='sep_conv2d_2')
                x5 = tf.add(x5_1, x5_2, name='normal_add')

            out_x = tf.concat([prev_x, x1, x2, x3, x4, x5], axis=-1, name='normal_concat')  # default: x = tf.concat([x1, x2, x3, x4, x5], axis=-1, name='normal_concat')

        return out_x, x

    def reduction_cell(self, x, prev_x, num_filters, name='reduction_cell'):
        with tf.variable_scope(name_or_scope=name):
            prev_x = self.adjust_block(x=x, prev_x=prev_x, num_filters=num_filters, name='adjust_block')

            h = tf.nn.relu(features=x)
            h = self.conv2d(inputs=h, filters=num_filters, name='reduction_conv')
            h = self.batch_norm(inputs=h, act=None, name='reduction_batch_norm')

            with tf.variable_scope(name_or_scope='block_1'):
                x1_1 = tf.nn.avg_pool(value=h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='avg_pool')
                x1_2 = self.separable_conv2d_block(inputs=prev_x, kernel_size=5, filters=num_filters, strides=2, name='sep_conv2d')
                x1 = tf.add(x1_1, x1_2, name='reduction_add')

            with tf.variable_scope(name_or_scope='block_2'):
                x2_1 = tf.nn.max_pool(value=h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool')
                x2_2 = self.separable_conv2d_block(inputs=prev_x, kernel_size=7, filters=num_filters, strides=2, name='sep_conv2d')
                x2 = tf.add(x2_1, x2_2, name='reduction_add')

            with tf.variable_scope(name_or_scope='block_3'):
                x3_1 = self.separable_conv2d_block(inputs=prev_x, kernel_size=7, filters=num_filters, strides=2, name='sep_conv2d_1')
                x3_2 = self.separable_conv2d_block(inputs=h, kernel_size=5, filters=num_filters, strides=2, name='sep_conv2d_2')
                x3 = tf.add(x3_1, x3_2, name='reduction_add')

            with tf.variable_scope(name_or_scope='block_4'):
                x4 = tf.nn.avg_pool(value=x3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='avg_pool')
                x4 = tf.add(x4, x2, name='reduction_add')

            with tf.variable_scope(name_or_scope='block_5'):
                x5_1 = tf.nn.max_pool(value=h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool')
                x5_2 = self.separable_conv2d_block(inputs=x3, filters=num_filters, strides=1, name='sep_conv2d')
                x5 = tf.add(x5_1, x5_2, name='reduction_add')

            out_x = tf.concat([x1, x2, x4, x5], axis=-1, name='reduction_concat')  # default: x = tf.concat([x1, x4, x5], axis=-1, name='reduction_concat')

        return out_x, x

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

    def separable_conv2d_block(self, inputs, filters, kernel_size=3, strides=1, padding='SAME', depth_multiplier=1, name='separable_conv2d_block'):
        with tf.variable_scope(name_or_scope=name):
            x = tf.nn.relu(inputs)
            x = self.separable_conv2d(inputs=x, kernel_size=kernel_size, filters=filters, strides=strides, padding=padding,
                                      depth_multiplier=depth_multiplier, name='sep_conv_1')
            x = self.batch_norm(inputs=x, act=None, name='batch_norm_1')

            x = tf.nn.relu(x)
            x = self.separable_conv2d(inputs=x, kernel_size=kernel_size, filters=filters, padding=padding,
                                      depth_multiplier=depth_multiplier, name='sep_conv_2')
            x = self.batch_norm(inputs=x, act=None, name='batch_norm_2')

        return x

    def separable_conv2d(self, inputs, filters, kernel_size=3, strides=1, padding='SAME', depth_multiplier=1, name=None):
        return slim.separable_conv2d(inputs=inputs, num_outputs=filters, kernel_size=kernel_size, activation_fn=tf.identity,
                                     biases_initializer=None, weights_initializer=self.kernel_initializer, weights_regularizer=self.kernel_regularizer,
                                     depth_multiplier=depth_multiplier, stride=strides, padding=padding, scope=name)

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
            return self.sess.run([self.accuracy, self.loss, self.summary_merged_values, self.train_op], feed_dict={self.x: x, self.y: y, self.is_train: True})
        else:
            return self.sess.run([self.accuracy, self.loss, self.train_op], feed_dict={self.x: x, self.y: y, self.is_train: True})

    def test(self, x, y):
        return self.sess.run([self.accuracy, self.loss, self.prob], feed_dict={self.x: x, self.y: y, self.is_train: False})