import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

class Generator:
    def __init__(self, hidden_num, batch_size, s_size=(4, 4), channel=3, layer_num=6, output_size=[(12, 16), (24, 32), (48, 64), (96, 128), (192, 256)]):
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.s_size = s_size
        self.channel = channel
        self.layer_num = layer_num
        self.output_size = output_size
        self.carry = tf.placeholder(dtype=tf.float32, shape=None, name='carry')

    def __call__(self, inputs, training=True, reuse=False, name='Generator'):

        def Deconvolution(layer, h0_layer, idx, name):
            with tf.variable_scope(name_or_scope=name):
                conv2d_a = Conv2d(layer=layer, n_filter=self.hidden_num, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_a')
                conv2d_b = Conv2d(layer=conv2d_a, n_filter=self.hidden_num, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_b')

                if self.training:
                    width, height = self.s_size if idx == 0 else self.output_size[idx-1]
                    conv2d_b.outputs = tf.reshape(tf.add(tf.multiply(self.carry, conv2d_a.outputs), tf.multiply((1. - self.carry), conv2d_b.outputs)),
                                                  shape=(-1, width, height, self.hidden_num))

                if idx == (self.layer_num - 1):
                    return conv2d_b

                conv2d_b.outputs = tf.concat([tf.image.resize_nearest_neighbor(conv2d_b.outputs, self.output_size[idx], name='conv_resize'),
                                              tf.image.resize_nearest_neighbor(h0_layer.outputs, self.output_size[idx], name='h0_resize')], -1)
            return conv2d_b

        self.training = training
        self.reuse = reuse

        with tf.variable_scope(name_or_scope=name, reuse=self.reuse):

            with tf.variable_scope(name_or_scope='input'):
                layer = InputLayer(inputs=inputs, name='input')
                layer = DenseLayer(prev_layer=layer, n_units=self.hidden_num * self.s_size[0] * self.s_size[1], name='dense')
                h0_layer = ReshapeLayer(prev_layer=layer, shape=(-1, self.s_size[0], self.s_size[1], self.hidden_num), name='reshape')

            with tf.variable_scope(name_or_scope='deconv'):
                for idx in range(self.layer_num):
                    layer = Deconvolution(layer=h0_layer if idx == 0 else layer, h0_layer=h0_layer, idx=idx, name='deconv_' + str(idx))

                # h0_layer.outputs = tf.image.resize_nearest_neighbor(h0_layer.outputs, self.output_size[-1], name='resize')
                # layer = ConcatLayer([layer, h0_layer], name='concat_layer')

            with tf.variable_scope('output'):
                layer = Conv2d(layer=layer, n_filter=self.channel, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='g_logit')

        self.variables = tl.layers.get_variables_with_name(name, True, True)

        return layer.outputs

class Discriminator:
    def __init__(self, hidden_num, batch_size, s_size=(4, 4), channel=3, layer_num=6, z_dim=100, output_size=[(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]):
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.s_size = s_size
        self.channel = channel
        self.layer_num = layer_num
        self.z_dim = z_dim
        self.output_size = output_size
        self.carry = tf.placeholder(dtype=tf.float32, shape=None, name='carry')

    def __call__(self, inputs, training=True, reuse=False, name='Discriminator'):

        def Deconvolution(layer, h0_layer, idx, name):
            with tf.variable_scope(name_or_scope=name):
                conv2d_a = Conv2d(layer=layer, n_filter=self.hidden_num, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_a')
                conv2d_b = Conv2d(layer=conv2d_a, n_filter=self.hidden_num, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_b')

                if self.training:
                    width, height = self.s_size if idx == 0 else self.output_size[idx - 1]
                    conv2d_b.outputs = tf.reshape(tf.add(tf.multiply(self.carry, conv2d_a.outputs), tf.multiply((1. - self.carry), conv2d_b.outputs)),
                                                  shape=(-1, width, height, self.hidden_num))

                if idx == (self.layer_num - 1):
                    return conv2d_b

                conv2d_b.outputs = tf.concat([tf.image.resize_nearest_neighbor(conv2d_b.outputs, self.output_size[idx], name='conv_resize'),
                                              tf.image.resize_nearest_neighbor(h0_layer.outputs, self.output_size[idx], name='h0_resize')], -1)
            return conv2d_b

        self.training = training
        self.reuse = reuse

        with tf.variable_scope(name_or_scope=name, reuse=self.reuse):
            layer = InputLayer(inputs=inputs, name='e_input')

            ## Encoder
            with tf.variable_scope(name_or_scope='encoder'):
                for idx in range(self.layer_num):
                    with tf.variable_scope(name_or_scope='conv2d_' + str(idx)):
                        conv2d_a = Conv2d(layer=layer, n_filter=self.hidden_num * (idx + 1), filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_a')
                        conv2d_b = Conv2d(layer=conv2d_a, n_filter=self.hidden_num * (idx + 1), filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_b')

                        if self.training:
                            width, height = self.s_size if idx == (self.layer_num - 1) else self.output_size[self.layer_num - idx - 2]
                            conv2d_b.outputs = tf.reshape(tf.add(tf.multiply(self.carry, conv2d_a.outputs), (1. - self.carry) * conv2d_b.outputs),
                                                          shape=(-1, width, height, self.hidden_num * (idx + 1)))

                        if idx < (self.layer_num - 1):
                            layer = Conv2d(layer=conv2d_b, n_filter=self.hidden_num * (idx + 2), filter_size=(3, 3), strides=(2, 2), act=tf.nn.elu, name='conv2d_c')

                layer = ReshapeLayer(prev_layer=conv2d_b, shape=(-1, self.s_size[0] * self.s_size[1] * self.hidden_num * self.layer_num), name='e_reshape')

                z = DenseLayer(prev_layer=layer, n_units=self.z_dim, name='z_dense')
                z = tf.nn.tanh(z.outputs, name='z_logit')

            layer = DenseLayer(prev_layer=layer, n_units=self.s_size[0] * self.s_size[1] * self.hidden_num, name='d_input')
            h0_layer = ReshapeLayer(prev_layer=layer, shape=(-1, self.s_size[0], self.s_size[1], self.hidden_num), name='d_reshape')

            ## Decoder
            with tf.variable_scope(name_or_scope='decoder'):
                for idx in range(self.layer_num):
                    with tf.variable_scope(name_or_scope='deconv'):
                        layer = Deconvolution(layer=h0_layer if idx == 0 else layer, h0_layer=h0_layer, idx=idx, name='deconv_' + str(idx))
                # h0_layer.outputs = tf.image.resize_nearest_neighbor(h0_layer.outputs, self.output_size[-1], name='resize')
                # layer = ConcatLayer([layer, h0_layer], name='concat_layer')

            with tf.variable_scope('output'):
                layer = Conv2d(layer=layer, n_filter=self.channel, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='d_logit')

        self.variables = tl.layers.get_variables_with_name(name, True, True)

        return layer.outputs, z

class BEGAN:
    def __init__(self, sess, batch_size=16, z_dim=64, img_size=(128, 128), learning_rate=0.0001, channel=3, layer_num=5,
                 hidden_num=80, output_size=[(16, 16), (32, 32), (64, 64), (128, 128)]):
        self.sess = sess
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.img_size = img_size
        self.channel = channel
        self.layer_num = layer_num
        self.learning_rate = learning_rate
        self.gamma = 0.5
        self.lambda_k = 0.001

        self.g = Generator(hidden_num=hidden_num, batch_size=self.batch_size, s_size=(8, 8), channel=self.channel, layer_num=self.layer_num, output_size=output_size)
        self.d = Discriminator(hidden_num=hidden_num, batch_size=self.batch_size, s_size=(8, 8), channel=self.channel, layer_num=self.layer_num, z_dim=z_dim, output_size=output_size)
        self._network()

    def _network(self):
        with tf.variable_scope('input_data'):
            self.z = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.z_dim), name='z_data')
            self.x = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size[0], self.img_size[1], self.channel), name='x_data')
            self.k_t = tf.Variable(0., trainable=False, dtype=tf.float32, name='k_t')

        with tf.variable_scope('network'):
            self.g_digit = self.g(inputs=self.z, training=True, reuse=False, name='Generator')
            self.AE_G, self.G_d_z = self.d(inputs=self.g_digit, training=True, reuse=False, name='Discriminator')
            self.AE_X, self.X_d_z = self.d(inputs=self.x, training=True, reuse=True, name='Discriminator')
            # self.AE_G = self.d(inputs=self.g_digit, training=True, reuse=False, name='Discriminator')
            # self.AE_X = self.d(inputs=self.x, training=True, reuse=True, name='Discriminator')

            self.g_img = self._denorm_img(self.g_digit)
            self.AE_G_img, self.AE_X_img = self._denorm_img(self.AE_G), self._denorm_img(self.AE_X)
            # self.g_img = self.g_digit
            # self.AE_G_img, self.AE_X_img = self.AE_G, self.AE_X

            self.g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.d_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.d_loss_real = tf.reduce_mean(tf.abs(self.x - self.AE_X))
            self.d_loss_fake = tf.reduce_mean(tf.abs(self.g_digit - self.AE_G))

            self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
            self.g_loss = self.d_loss_fake

            self.d_opt = self.d_opt.minimize(self.d_loss, var_list=self.d.variables)
            self.g_opt = self.g_opt.minimize(self.g_loss, var_list=self.g.variables)

            self.balance = self.gamma * self.d_loss_real - self.g_loss
            self.measure = self.d_loss_real + tf.abs(self.balance)

            with tf.control_dependencies([self.d_opt, self.g_opt]):
                self.k_update = tf.assign(self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

    def _denorm_img(self, values):
        return tf.clip_by_value(t=tf.multiply(tf.add(values, 1), 127.5), clip_value_min=0, clip_value_max=255)

    def train(self, z, x, carry):
        self.g.training, self.d.training = True, True
        return self.sess.run([self.d_loss, self.g_loss, self.k_t, self.measure, self.k_update],
                             feed_dict={self.z: z, self.x: x, self.g.carry: carry, self.d.carry: carry})

    def generate(self, z):
        self.g.training = False
        return self.sess.run(self.g_img, feed_dict={self.z: z, self.g.carry: 0})
#
# import numpy as np
#
# with tf.Session() as sess:
#     x = tf.placeholder(dtype=tf.float32, shape=[None, 30, 30, 3], name='x')
#     carry = tf.placeholder(dtype=tf.float32, shape=None, name='carry')
#
#     layer = InputLayer(inputs=x)
#     conv2d_a = Conv2d(layer=layer, n_filter=30, filter_size=(3, 3), name='conv2d_a')
#     conv2d_b = Conv2d(layer=conv2d_a, n_filter=30, filter_size=(3, 3), name='conv2d_b')
#     conv2d_b.outputs = tf.reshape(tf.add(tf.multiply(carry, conv2d_a.outputs), tf.multiply((1. - carry), conv2d_b.outputs)),
#                                            shape=(5, 30, 30, 30))
#     conv2d_b.outputs = tf.image.resize_nearest_neighbor(conv2d_b.outputs, (40, 40), name='resize')
#     conv2d_c = Conv2d(layer=conv2d_b, n_filter=30, filter_size=(3, 3), name='conv2d_c')
#
#     sess.run(tf.global_variables_initializer())
#     c = sess.run(conv2d_c.outputs, feed_dict={x: np.random.normal(size=(5, 30, 30, 3)), carry: 1})
#     print(c.shape)