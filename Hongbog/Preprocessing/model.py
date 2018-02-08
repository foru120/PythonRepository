import tensorflow as tf
from tensorflow.contrib.layers import *

class Model:
    def __init__(self, sess, depth):
        self.sess = sess
        self.N = int((depth - 4) / 3)  # layer 개수
        self.growthRate = 12  # k
        self.compression_factor = 0.5
        self._build_graph()

    def _build_graph(self):
        with tf.name_scope('initialize_scope'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 1], name='X_data')
            self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_data')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
            self.learning_rate = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
            self.regularizer = l2_regularizer(1e-3)

        def conv(l, kernel, channel, stride):
            return conv2d(inputs=l, num_outputs=channel, kernel_size=kernel, stride=stride, padding='SAME', activation_fn=None,
                          weights_initializer=variance_scaling_initializer(), biases_initializer=None, weights_regularizer=self.regularizer)

        def _depthwise_separable_conv(layer, kernel, output, downsample=False, wm=1., scope=None):
            _stride = [2, 2] if downsample else [1, 1]
            with tf.contrib.framework.arg_scope(
                    [batch_norm],
                    activation_fn=tf.nn.elu,
                    decay=0.99,
                    updates_collections=None,
                    scale=True,
                    is_training=self.training):
                    layer = separable_conv2d(inputs=layer, num_outputs=None, kernel_size=kernel,
                                             stride=_stride, depth_multiplier=wm, padding='SAME',
                                             weights_regularizer=self.regularizer * 1e-4,
                                             activation_fn=None, scope='depthwise_conv')
                    layer = batch_norm(inputs=layer, scope='dw_batch_norm')
                    layer = conv2d(inputs=layer, num_outputs=output * wm, kernel_size=1,
                                   weights_regularizer=self.regularizer, activation_fn=None, scope='pointwise_conv')
                    layer = batch_norm(inputs=layer, scope='pw_batch_norm')
            return layer

        def add_layer(name, l):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                c = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                c = tf.nn.elu(c, 'bottleneck')
                c = conv(c, 1, 4 * self.growthRate, 1)  # 4k, output
                c = dropout(inputs=c, keep_prob=self.dropout_rate, is_training=self.training)

                '''basic dense layer'''
                c = batch_norm(inputs=c, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                c = tf.nn.elu(c, 'basic_1')
                c = _depthwise_separable_conv(c, [3, 3], self.growthRate)
                # c = conv(c, 3, self.growthRate, 1)  # k, output
                c = dropout(inputs=c, keep_prob=self.dropout_rate, is_training=self.training)

                l = tf.concat([c, l], axis=3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name):
                '''compression transition layer (DenseNet-C)'''
                l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
                l = tf.nn.elu(l, 'transition')
                l = conv(l, 1, int(in_channel * self.compression_factor), 1)
                l = avg_pool2d(inputs=l, kernel_size=[2, 2], stride=2, padding='SAME')
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            return l

        def dense_net():
            l = conv(self.X, 5, 16, 1)

            with tf.variable_scope('dense_block1'):
                for i in range(self.N):
                    l = add_layer('dense_layer_{}'.format(i), l)
                l = add_transition('transition1', l)

            with tf.variable_scope('dense_block2'):
                for i in range(self.N):
                    l = add_layer('dense_layer_{}'.format(i), l)
                l = add_transition('transition2', l)

            with tf.variable_scope('dense_block3'):
                for i in range(self.N):
                    l = add_layer('dense_layer_{}'.format(i), l)

            l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
            l = tf.nn.elu(l, 'output')
            l = avg_pool2d(inputs=l, kernel_size=[4, 4], stride=1, padding='VALID')
            l = tf.reshape(l, shape=[-1, 1 * 1 * 256])
            # l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            logits = fully_connected(inputs=l, num_outputs=2, activation_fn=None,
                                     weights_initializer=variance_scaling_initializer(), weights_regularizer=l2_regularizer(1e-3))

            return logits

        self.logits = dense_net()
        self.prob = tf.nn.softmax(logits=self.logits, name='output')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        loss = tf.reduce_mean(loss, name='cross_entropy_loss')
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([loss] + reg_losses, name='loss')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False, self.dropout_rate: 1.0})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.y: y_test, self.training: False, self.dropout_rate: 1.0})

    def train(self, x_data, y_data):
        return self.sess.run([self.accuracy, self.optimizer], feed_dict={self.X: x_data, self.y: y_data,
                                                                                    self.training: True, self.dropout_rate: 0.8})

    def validation(self, x_test, y_test):
        return self.sess.run([self.loss, self.accuracy], feed_dict={self.X: x_test, self.y: y_test, self.training: False, self.dropout_rate: 1.0})

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name+'alphas', _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x, name=name+'p_relu')
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg