import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

class Model:
    def __init__(self, sess):
        self.sess = sess
        self.N = 12  # Dense Block 내의 Layer 개수
        self.growthRate = 24  # k
        self.compression_factor = 0.5
        self._build_graph()

    def _build_graph(self):
        with tf.name_scope('initialize_scope'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 180, 60, 1], name='X_data')
            self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_data')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
            self.learning_rate = tf.get_variable('learning_rate', initializer=0.1, trainable=False)

        def BatchNorm(layer, name):
            with tf.variable_scope(name_or_scope=name):
                return BatchNormLayer(layer, act=tf.nn.elu, is_train=tf.where(tf.equal(self.training, True), True, False))

        def _add_layer(input, name):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                layer = BatchNorm(layer=input, name='bottleneck_batch_norm')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, name='bottleneck_dropout')
                layer = Conv2d(layer=layer, n_filter=4 * self.growthRate, filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='bottleneck_conv')

                '''basic dense layer'''
                layer = BatchNorm(layer=layer, name='basic_batch_norm')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, name='basic_dropout_a')
                layer = DepthwiseConv2d(prev_layer=layer, shape=(3, 3), strides=(1, 1), act=tf.nn.elu, name='basic_dwconv')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, name='basic_dropout_b')

                layer.outputs = tf.concat([layer.outputs, input.outputs], axis=3)
                skip_layer = Conv2d(layer=input, n_filter=layer.outputs.get_shape()[-1], filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='basic_skip')
                layer = layer.outputs + skip_layer.outputs

            return layer

        def _add_transition(layer, name):
            with tf.variable_scope(name):
                '''compression transition layer (DenseNet-C)'''
                layer = BatchNorm(layer=layer, name='trasition_batch_norm')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, name='transition_dropout_a')
                layer = Conv2d(layer=layer, n_filter=layer.outputs.get_shape()[-1], filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='transition_conv')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.avg_pool, name='transition_pool')
            return layer

        def dense_net():
            layer = InputLayer(inputs=self.X, name='input')
            layer = Conv2d(layer=layer, n_filter=16, filter_size=(5, 5), strides=(1, 1), act=tf.identity, name='input_conv')

            with tf.variable_scope('dense_block1'):
                for i in range(self.N):
                    layer = _add_layer(input=layer, name='dense_layer_{}'.format(i))
                    layer = _add_transition('transition1', layer)

            with tf.variable_scope('dense_block2'):
                for i in range(self.N):
                    layer = _add_layer(input=layer, name='dense_layer_{}'.format(i))
                    layer = _add_transition('transition2', layer)

            with tf.variable_scope('dense_block3'):
                for i in range(self.N):
                    layer = _add_layer(input=layer, name='dense_layer_{}'.format(i))

            with tf.variable_scope('output'):
                layer = BatchNorm(layer=layer, name='output_batch_norm')
                layer = PoolLayer(prev_layer=layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), pool=tf.nn.avg_pool, name='output_pool')
                layer = ReshapeLayer(prev_layer=layer, shape=(-1, 1 * 1 * 508), name='output_reshape')
                layer = DropoutLayer(prev_layer=layer, keep=self.dropout_rate, is_train=self.training, name='output_dropout')
                layer = DenseLayer(prev_layer=layer, n_units=2, act=tf.identity, name='logit')

            return layer

        self.logits = dense_net()
        self.prob = tf.nn.softmax(logits=self.logits, name='softmax')
        self.loss = tl.cost.cross_entropy(output=self.logits, target=self.y, name='ce_loss')
        regularizer = tf.contrib.layers.l2_regularizer(0.001)
        self.loss = self.loss + regularizer(self.logits.all_params[0]) + regularizer(self.logits.all_params[2])

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float64))

    def predict(self, x_test):
        return self.sess.run(self.prob, feed_dict={self.X: x_test, self.training: False, self.dropout_rate: 1.0})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test, self.y: y_test, self.training: False, self.dropout_rate: 1.0})

    def train(self, x_data, y_data):
        return self.sess.run([self.accuracy, self.loss, self.optimizer], feed_dict={self.X: x_data, self.y: y_data,
                                                                                    self.training: True,
                                                                                    self.dropout_rate: 0.8})

    def validation(self, x_test, y_test):
        return self.sess.run([self.accuracy, self.loss],
                             feed_dict={self.X: x_test, self.y: y_test, self.training: False, self.dropout_rate: 1.0})