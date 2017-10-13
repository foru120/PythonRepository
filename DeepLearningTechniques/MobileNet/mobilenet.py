import tensorflow as tf

class Model:
    def __init__(self, sess, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate
        self._build_graph()

    def _depthwise_separable_conv(self, layer, output, downsample=False, wm=1., scope=None):
        _stride = [2, 2] if downsample else [1, 1]
        with tf.variable_scope(scope):
            layer = tf.contrib.layers.separable_conv2d(inputs=layer, num_outputs=None, kernel_size=[3, 3], stride=_stride,
                                                       depth_multiplier=wm, padding='SAME', weights_regularizer=self.depthwise_regularizer, scope='depthwise_conv')
            layer = tf.contrib.layers.batch_norm(inputs=layer, scope='dw_batch_norm')
            layer = tf.contrib.layers.conv2d(inputs=layer, num_outputs=output * wm, kernel_size=1, weights_regularizer=self.etc_regularizer, scope='pointwise_conv')
            layer = tf.contrib.layers.batch_norm(inputs=layer, scope='pw_batch_norm')
            layer = tf.contrib.layers.dropout(inputs=layer, keep_prob=self.dropout_rate, is_training=self.training)
            # layer = tf.nn.dropout(layer, self.dropout_rate, name='dropout')
        return layer

    def _build_graph(self):
        with tf.name_scope('initialize_step'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
            self.y = tf.placeholder(dtype=tf.int64, shape=[None])
            self.training = tf.placeholder(dtype=tf.bool, shape=None, name='training')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=None, name='dropout_rate')
            self.weight_initializer = tf.contrib.layers.variance_scaling_initializer()
            self.depthwise_regularizer = tf.contrib.layers.l2_regularizer(1e-8)
            self.etc_regularizer = tf.contrib.layers.l2_regularizer(1e-3)

        with tf.variable_scope('mobile_net'):
            with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.conv2d, tf.contrib.layers.separable_conv2d],
                    activation_fn=None,
                    biases_initializer=None,
                    weights_initializer=self.weight_initializer):
                with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.batch_norm],
                        activation_fn=tf.nn.elu,
                        decay=0.99,
                        updates_collections=None,
                        scale=True,
                        is_training=self.training):
                    layer = tf.contrib.layers.conv2d(inputs=self.X, num_outputs=40, kernel_size=3, stride=1, weights_regularizer=self.etc_regularizer, scope='conv_1')
                    layer = tf.contrib.layers.batch_norm(inputs=layer, scope='conv_1/batch_norm')
                    layer = self._depthwise_separable_conv(layer, 80, downsample=True, scope='conv_ds_2')
                    layer = self._depthwise_separable_conv(layer, 80, scope='conv_ds_3')
                    # layer = self._depthwise_separable_conv(layer, 80, downsample=True, scope='conv_ds_4')
                    # layer = self._depthwise_separable_conv(layer, 40, scope='conv_ds_4')
                    layer = self._depthwise_separable_conv(layer, 160, downsample=True, scope='conv_ds_4')
                    layer = self._depthwise_separable_conv(layer, 160, scope='conv_ds_5')
                    for idx in range(6, 8):
                        layer = self._depthwise_separable_conv(layer, 320, scope='conv_ds_' + str(idx))
                    layer = self._depthwise_separable_conv(layer, 640, scope='conv_ds_8')
                    # layer = self._depthwise_separable_conv(layer, 400, scope='conv_ds_14')
                    layer = tf.contrib.layers.avg_pool2d(inputs=layer, kernel_size=[8, 8], stride=1, padding='VALID')
                    layer = tf.reshape(layer, shape=[-1, 1 * 1 * 640])
                    self.logits = tf.contrib.layers.fully_connected(inputs=layer, num_outputs=10, activation_fn=None,
                                                                    weights_initializer=self.weight_initializer, weights_regularizer=self.etc_regularizer)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y), name='cross_entropy_loss')
                    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    self.loss = tf.add_n([loss] + reg_losses, name='loss')
                    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float32))

    def train(self, x, y):
        return self.sess.run([self.accuracy, self.optimizer], feed_dict={self.X: x, self.y: y, self.training: True, self.dropout_rate: 0.6})

    def validation(self, x, y):
        return self.sess.run([self.loss, self.accuracy], feed_dict={self.X: x, self.y: y, self.training: False, self.dropout_rate: 1.0})

    def get_accuracy(self, x, y):
        return self.sess.run([self.accuracy], feed_dict={self.X: x, self.y: y, self.training: False, self.dropout_rate: 1.0})