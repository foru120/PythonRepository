import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.contrib.layers import batch_norm, variance_scaling_initializer

class LSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state
            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh', [x_size, 4 * self.num_units], initializer=variance_scaling_initializer())
            W_hh = tf.get_variable('W_hh', [self.num_units, 4 * self.num_units], initializer=variance_scaling_initializer())
            bias = tf.get_variable('bias', [4 * self.num_units])

            # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
            # improve speed by concat.
            concat = tf.concat(1, [x, h])
            W_both = tf.concat(0, [W_xh, W_hh])
            hidden = tf.matmul(concat, W_both) + bias
            i, g, f, o = tf.split(hidden, 4, 1)
            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(g)
            new_h = tf.nn.softsign(new_c) * tf.sigmoid(o)
            return new_h, (new_c, new_h)

class BNLSTMCell(RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''

    def __init__(self, num_units, training):
        self.num_units = num_units
        self.training = training

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh', [x_size, 4 * self.num_units], initializer=variance_scaling_initializer())

            W_hh = tf.get_variable('W_hh', [self.num_units, 4 * self.num_units], initializer=variance_scaling_initializer())
            bias = tf.get_variable('bias', [4 * self.num_units])

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)
            bn_xh = bn_rnn(xh, 'xh', self.training)
            bn_hh = bn_rnn(hh, 'hh', self.training)
            hidden = bn_xh + bn_hh + bias

            i, g, f, o = tf.split(hidden, 4, 1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(g)
            bn_new_c = bn_rnn(new_c, 'c', self.training)
            new_h = tf.nn.softsign(bn_new_c) * tf.sigmoid(o)
            # new_h -> 활성화함수(input) * o

            return new_h, (new_c, new_h)

def bn_rnn(x, name_scope, training, epsilon=1e-3, decay=0.999):
    '''Assume 2d [batch, values] tensor'''
    with tf.variable_scope(name_scope):
        return batch_norm(inputs=x, scale=True, epsilon=epsilon, decay=decay, updates_collections=None, is_training=training)
