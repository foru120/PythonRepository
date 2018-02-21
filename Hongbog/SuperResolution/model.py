import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *

class Model:
    def __init__(self, sess):
        self.sess = sess
        self.N = 12  # Dense Block 내의 Layer 개수
        self.growthRate = 24  # K
        self.compression_factor = 0.5
        self._build_graph()

    def _build_graph(self):
        with tf.name_scope('initialize_scope'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='X_data')
            self.y = tf.placeholder(dtype=tf.int64, shape=[None, 64, 64, 1], name='y_data')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.learning_rate = tf.get_variable('learning_rate', initializer=1e-5, trainable=False)
            self.regularizer = l2_regularizer(1e-3)

        def _conv(l, kernel, channel, stride, scope):
            return conv2d(inputs=l, num_outputs=channel, kernel_size=kernel, stride=stride, padding='SAME', activation_fn=None,
                          weights_initializer=variance_scaling_initializer(), biases_initializer=None, weights_regularizer=self.regularizer,
                          scope=scope)

        def _batch_norm(inputs, scope):
            with tf.contrib.framework.arg_scope(
                [batch_norm],
                activation_fn=None,
                decay=0.99,
                updates_collections=None,
                scale=True,
                zero_debias_moving_mean=True,
                is_training=self.training,
                scope=scope):
                return batch_norm(inputs=inputs)

        def _depthwise_separable_conv(layer, kernel, output, downsample=False, wm=1., scope=None):
            _stride = [2, 2] if downsample else [1, 1]
            layer = separable_conv2d(inputs=layer, num_outputs=None, kernel_size=kernel,
                                     stride=_stride, depth_multiplier=wm, padding='SAME',
                                     weights_regularizer=l2_regularizer(1e-4),
                                     activation_fn=None, scope='depthwise_conv')
            layer = _batch_norm(inputs=layer, scope='dw_batch_norm')
            layer = tf.nn.elu(layer, 'dw_elu')
            # layer = tf.nn.softplus(layer, 'dw_softplus')
            layer = conv2d(inputs=layer, num_outputs=output * wm, kernel_size=1,
                           weights_regularizer=self.regularizer, activation_fn=None, scope='pointwise_conv')
            layer = _batch_norm(inputs=layer, scope='pw_batch_norm')
            layer = tf.nn.elu(layer, 'pw_elu')
            # layer = tf.nn.softplus(layer, 'pw_softplus')

            return layer

        def _add_layer(name, l):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                c = _batch_norm(l, 'bottleneck_batch_norm')
                c = tf.nn.elu(c, 'bottleneck')
                # c = tf.nn.softplus(c, 'bottleneck')
                c = _conv(c, 1, 4 * self.growthRate, 1)  # 4k, output

                '''basic dense layer'''
                c = _batch_norm(inputs=c, scope='basic_batch_norm')
                c = tf.nn.elu(c, 'basic_1')
                # c = tf.nn.softplus(c, 'basic_1')

                # c = _depthwise_separable_conv(c, [3, 3], self.growthRate)
                c = _conv(c, 3, self.growthRate, 1)  # k, output

                l = tf.concat([c, l], axis=3)
            return l

        def dense_net():
            l = _conv(self.X, 3, 2*self.growthRate, 1, 'first_layer')

            with tf.variable_scope('dense_block1'):
                for i in range(self.N):
                    l = _add_layer('dense_layer_{}'.format(i), l)

            with tf.variable_scope('dense_block2'):
                for i in range(self.N):
                    l = _add_layer('dense_layer_{}'.format(i), l)

            with tf.variable_scope('dense_block3'):
                for i in range(self.N):
                    l = _add_layer('dense_layer_{}'.format(i), l)

            with tf.variable_scope('dense_block4'):
                for i in range(self.N):
                    l = _add_layer('dense_layer_{}'.format(i), l)

            l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
            l = tf.nn.elu(l, 'output')
            # l = tf.nn.softplus(l, 'output')
            logits = _conv(l, 3, 1, 1, 'output_layer')

            return logits

        self.logits = dense_net()
        self.prob = tf.nn.softmax(logits=self.logits, name='output')
        self.mse = tf.losses.mean_squared_error(self.y, self.logits, scope='mean_square_error')
        # loss = tf.reduce_mean(loss, name='cross_entropy_loss')
        self.loss = tf.add_n([self.mse] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.psnr = 20*math.log10(255.0/self.mse)

    def predict(self, x_test):
        return self.sess.run(self.prob, feed_dict={self.X: x_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.psnr, self.loss, self.optimizer], feed_dict={self.X: x_data, self.y: y_data,
                                                                                    self.training: True})
    def validation(self, x_test, y_test):
        return self.sess.run([self.loss, self.psnr], feed_dict={self.X: x_test, self.y: y_test, self.training: False})