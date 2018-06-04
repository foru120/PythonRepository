import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *

class Model:
    def __init__(self, sess):
        self.sess = sess
        self.N = 12  # Dense Block 내의 Layer 개수
        self.growthRate = 24  # k
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

        def _conv(l, kernel, channel, stride):
            return conv2d(inputs=l, num_outputs=channel, kernel_size=kernel, stride=stride, padding='SAME',
                          activation_fn=None, weights_initializer=variance_scaling_initializer(),
                          biases_initializer=None, weights_regularizer=self.regularizer)

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

        def _depthwise_separable_conv(layer, kernel, output, downsample=False, wm=1.):
            _stride = [2, 2] if downsample else [1, 1]
            layer = separable_conv2d(inputs=layer, num_outputs=None, kernel_size=kernel,
                                     stride=_stride, depth_multiplier=wm, padding='SAME',
                                     weights_regularizer=l2_regularizer(1e-4),
                                     activation_fn=None, scope='depthwise_conv')
            layer = _batch_norm(inputs=layer, scope='dw_batch_norm')
            layer = tf.nn.elu(layer, 'dw_elu')

            layer = conv2d(inputs=layer, num_outputs=output * wm, kernel_size=1,
                           weights_regularizer=self.regularizer, activation_fn=None, scope='pointwise_conv')
            layer = _batch_norm(inputs=layer, scope='pw_batch_norm')
            layer = tf.nn.elu(layer, 'pw_elu')

            return layer

        def _dense_block(name, l):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                c = _batch_norm(l, 'bottleneck_batch_norm')
                c = tf.nn.elu(c, 'bottleneck')
                c = dropout(inputs=c, keep_prob=self.dropout_rate, is_training=self.training)

                c = _conv(c, 1, 4 * self.growthRate, 1)  # 4k, output

                '''basic dense layer'''
                c = _batch_norm(inputs=c, scope='basic_batch_norm')
                c = tf.nn.elu(c, 'basic_1')
                c = dropout(inputs=c, keep_prob=self.dropout_rate, is_training=self.training)

                c = _depthwise_separable_conv(c, [3, 3], self.growthRate)
                c = dropout(inputs=c, keep_prob=self.dropout_rate, is_training=self.training)

                l = tf.concat([c, l], axis=3)
            return l

        def _transition_layer(name, l):
            with tf.variable_scope(name):
                '''compression transition layer (DenseNet-C)'''
                l = _batch_norm(inputs=l, scope='trasition_batch_norm')
                l = tf.nn.elu(l, 'transition')
                l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
                shape = l.get_shape().as_list()
                in_channel = shape[3]
                l = _conv(l, 1, int(in_channel * self.compression_factor), 1)
                l = avg_pool2d(inputs=l, kernel_size=[2, 2], stride=2, padding='SAME')
            return l

        def _network():
            l = _conv(self.X, 5, 16, 1)

            with tf.variable_scope('dense_block1'):
                for i in range(self.N):
                    l = _dense_block('dense_layer_{}'.format(i), l)
                l = _transition_layer('transition1', l)

            with tf.variable_scope('dense_block2'):
                for i in range(self.N):
                    l = _dense_block('dense_layer_{}'.format(i), l)
                l = _transition_layer('transition2', l)

            with tf.variable_scope('dense_block3'):
                for i in range(self.N):
                    l = _dense_block('dense_layer_{}'.format(i), l)

            l = batch_norm(inputs=l, decay=0.99, updates_collections=None, scale=True, is_training=self.training)
            l = tf.nn.elu(l, 'output')
            l = avg_pool2d(inputs=l, kernel_size=[4, 4], stride=1, padding='VALID')
            l = tf.reshape(l, shape=[-1, 1*1*508])  # k=12, shape=(-1,256)
            l = dropout(inputs=l, keep_prob=self.dropout_rate, is_training=self.training)
            logits = fully_connected(inputs=l, num_outputs=2, activation_fn=None,
                                     weights_initializer=variance_scaling_initializer(),
                                     weights_regularizer=self.regularizer)

            return logits

        self.logits = _network()
        self.prob = tf.nn.softmax(logits=self.logits, name='output')
        loss = self._focal_loss(alpha=0.3)
        self.loss = tf.add_n([loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='loss')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float64))

    def predict(self, x_test):
        return self.sess.run(self.prob, feed_dict={self.X: x_test, self.training: False, self.dropout_rate: 1.0})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.y: y_test, self.training: False,
                                                       self.dropout_rate: 1.0})

    def train(self, x_data, y_data):
        return self.sess.run([self.accuracy, self.loss, self.optimizer],
                             feed_dict={self.X: x_data, self.y: y_data, self.training: True, self.dropout_rate: 0.8})
    def validation(self, x_test, y_test):
        return self.sess.run([self.accuracy, self.loss],
                             feed_dict={self.X: x_test, self.y: y_test, self.training: False, self.dropout_rate: 1.0})

    def _softx_func(self, x, name):
        with tf.variable_scope(name):
            return (x / (1 - tf.exp(-x)))

    def _focal_loss(self, alpha=0.25, gamma=2):
        '''
        신경망에서 사용되는 loss 함수중의 하나 (데이터 셋의 비율이 한 쪽에 치우친 경우 사용하면 유용)
        :param alpha: positive 와 negative 의 loss 비율을 조정하는 하이퍼파라미터
        :param gamma: 실제 정답 레이블을 예측하는 확률 값의 크기를 조정하는 하이퍼파라미터
        :return: softmax 를 거쳐 나온 확률 값을 토대로 계산된 loss 값, type -> tensor
        '''
        zeros = array_ops.zeros_like(self.prob, dtype=self.prob.dtype)
        onehot_y = tf.one_hot(indices=self.y, depth=2, dtype=tf.float32)
        pos_p_sub = array_ops.where(onehot_y >= self.prob, onehot_y - self.prob, zeros)
        neg_p_sub = array_ops.where(onehot_y > zeros, zeros, self.prob)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(self.prob, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - self.prob, 1e-8, 1.0))
        return tf.reduce_mean(per_entry_cross_ent)

    def _parametric_relu(self, _x, name):
        alphas = tf.get_variable(name+'alphas', _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x, name=name+'p_relu')
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg