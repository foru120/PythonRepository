import tensorflow as tf
import tensorflow.contrib.slim as slim

class Model:
    def __init__(self, sess, scope):
        self.sess = sess
        self.scope = scope
        self.hidden_num = 16
        self._build_graph()

    def _build_graph(self):
        def _batch_norm(inputs, act=tf.nn.relu6, scope='batch_norm'):
            with slim.arg_scope([slim.batch_norm], decay=0.999, epsilon=0.001, reuse=False, zero_debias_moving_mean=True, scope=scope):
                return slim.batch_norm(inputs=inputs, activation_fn=act, is_training=self.training)

        def _residual_block(inputs, num_outputs, kernel_size, stride, padding='SAME', scope='residual_block'):
            with tf.variable_scope(scope):
                layer = _batch_norm(inputs=inputs, scope='batch_norm_1')
                layer = slim.conv2d(inputs=layer, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding, scope='residual_conv_1')
                layer = _batch_norm(inputs=layer, scope='batch_norm_2')
                layer = slim.conv2d(inputs=layer, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding, scope='residual_conv_2')
                layer = tf.add(inputs, layer, name='residual_add')
            return layer

        def _network():
            with tf.variable_scope(self.scope):
                with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                                    weights_initializer=self.initializer, weights_regularizer=self.regularizer, activation_fn=tf.identity):
                    with tf.variable_scope('input_network'):
                        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 60, 160, 1], name='x_data')
                        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_data')

                        self.training = tf.placeholder(dtype=tf.bool, shape=None, name='training')
                        self.initializer = tf.truncated_normal_initializer(stddev=0.01)
                        self.regularizer = slim.l2_regularizer(0.0005)

                    '''residual block'''
                    with tf.variable_scope('residual_block'):
                        for i in range(1, 20):
                            layer = _residual_block(inputs=self.x if i == 1 else layer, num_outputs=self.hidden_num, kernel_size=3,
                                                    stride=1, scope='residual_block_{}'.format(i))
                            if i % 4 == 0:
                                self.hidden_num *= 2
                                layer = slim.conv2d(inputs=layer, num_outputs=32, kernel_size=1, stride=1, scope='subsampling_{}'.format(int(i/4)))

                    self.cam_layer = layer

                    with tf.variable_scope('output_network'):
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

    def predict(self, x, training):
        return self.sess.run(self.prob,
                             feed_dict={self.x: x, self.training: training})

    def train(self, x, y, training):
        return self.sess.run([self.accuracy, self.loss, self.optimizer],
                             feed_dict={self.x: x, self.y: y, self.training: training})

    def validation(self, x, y, training):
        return self.sess.run([self.accuracy, self.loss, self.prob],
                             feed_dict={self.x: x, self.y: y, self.training: training})