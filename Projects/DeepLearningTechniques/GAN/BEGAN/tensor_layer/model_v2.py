import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

class Generator:
    def __init__(self, hidden_num, batch_size, s_size=(4, 4), channel=3, layer_num=6, output_size=[(12, 16), (24, 32), (48, 64), (96, 128), (192, 256)],
                 growthRate=10, n=3):
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.s_size = s_size
        self.channel = channel
        self.layer_num = layer_num
        self.output_size = output_size
        self.growthRate = growthRate  # Dense Network Filter's Growth Rate
        self.n = n  # Dense Block 내의 Layer 개수

    def __call__(self, inputs, training=True, reuse=False, name='Generator'):

        def _batch_norm(layer, act=tf.nn.elu, name='batch_norm'):
            with tf.variable_scope(name_or_scope=name):
                return BatchNormLayer(layer, act=act, is_train=self.training)

        def _dense_block(input, name):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                layer = _batch_norm(input, name='bottleneck_batch_norm')
                layer = Conv2d(layer=layer, n_filter=4 * self.growthRate, filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='bottleneck_conv')

                '''basic dense layer'''
                layer = _batch_norm(layer, name='basic_batch_norm')
                layer = Conv2d(layer=layer, n_filter=self.growthRate, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='basic_conv')

                layer = ConcatLayer([input, layer], name='concat')
            return layer

        def _deconvolution(layer, idx, name):
            with tf.variable_scope(name_or_scope=name):
                for i in range(self.n):
                    layer = _dense_block(layer, name='dense_block_{}'.format(i))

                if idx == (self.layer_num - 1):
                    return layer

                layer.outputs = tf.image.resize_nearest_neighbor(layer.outputs, self.output_size[idx], name='conv_resize')
            return layer

        self.training = training
        self.reuse = reuse

        with tf.variable_scope(name_or_scope=name, reuse=self.reuse):

            with tf.variable_scope(name_or_scope='input'):
                layer = InputLayer(inputs=inputs, name='input')
                layer = DenseLayer(prev_layer=layer, n_units=self.hidden_num * self.s_size[0] * self.s_size[1], name='dense')
                h0_layer = ReshapeLayer(prev_layer=layer, shape=(-1, self.s_size[0], self.s_size[1], self.hidden_num), name='reshape')

            with tf.variable_scope(name_or_scope='deconv'):
                for idx in range(self.layer_num):
                    layer = _deconvolution(layer=h0_layer if idx == 0 else layer, idx=idx, name='deconv_' + str(idx))
                layer.outputs = tf.concat([tf.image.resize_nearest_neighbor(h0_layer.outputs, self.output_size[-1], name='h0_resize'),
                                           layer.outputs], -1)

            with tf.variable_scope('output'):
                layer = Conv2d(layer=layer, n_filter=self.channel, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='g_logit')

        self.variables = tl.layers.get_variables_with_name(name, True, True)

        return layer.outputs

class Discriminator:
    def __init__(self, hidden_num, batch_size, s_size=(4, 4), channel=3, layer_num=6, z_dim=100, output_size=[(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)],
                 growthRate=10, n=3):
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.s_size = s_size
        self.channel = channel
        self.layer_num = layer_num
        self.z_dim = z_dim
        self.output_size = output_size
        self.growthRate = growthRate  # Dense Network Filter's Growth Rate
        self.n = n  # Dense Block 내의 Layer 개수

    def __call__(self, inputs, training=True, reuse=False, name='Discriminator'):

        def _batch_norm(layer, act=tf.nn.elu, name='batch_norm'):
            with tf.variable_scope(name_or_scope=name):
                return BatchNormLayer(layer, act=act, is_train=self.training)

        def _dense_block(input, name):
            with tf.variable_scope(name):
                '''bottleneck layer (DenseNet-B)'''
                layer = _batch_norm(input, name='bottleneck_batch_norm')
                layer = Conv2d(layer=layer, n_filter=4 * self.growthRate, filter_size=(1, 1), strides=(1, 1), act=tf.identity, name='bottleneck_conv')

                '''basic dense layer'''
                layer = _batch_norm(layer, name='basic_batch_norm')
                layer = Conv2d(layer=layer, n_filter=self.growthRate, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='basic_conv')

                layer = ConcatLayer([input, layer], name='concat')
            return layer

        def _deconvolution(layer, idx, name):
            with tf.variable_scope(name_or_scope=name):
                for i in range(self.n):
                    layer = _dense_block(layer, name='dense_block_{}'.format(i))

                if idx == (self.layer_num - 1):
                    return layer

                layer.outputs = tf.image.resize_nearest_neighbor(layer.outputs, self.output_size[idx], name='conv_resize')
            return layer

        def _residual_block(inputs, n_filter, name):
            '''
            Improved ResNet
             - original resnet 에서 마지막 단계의 summation 연산 시 ReLU 함수를 사용한 것과 달리 identity 함수를 사용해
               identity mapping 이 이루어지게 된다.
             - full pre-activation: batch normalization 을 residual net 상에서 activation 함수보다 앞으로 오는 구조.
            '''
            with tf.variable_scope(name):
                layer = _batch_norm(inputs, name='batch_norm_1')
                layer = Conv2d(layer=layer, n_filter=n_filter, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='residual_1')
                layer = _batch_norm(layer, name='batch_norm_2')
                layer = Conv2d(layer=layer, n_filter=n_filter, filter_size=(3, 3), strides=(1, 1), act=tf.identity, name='residual_2')
                layer.outputs = tf.add(inputs.outputs, layer.outputs)
            return layer

        self.training = training
        self.reuse = reuse

        with tf.variable_scope(name_or_scope=name, reuse=self.reuse):
            layer = InputLayer(inputs=inputs, name='e_input')
            layer = Conv2d(layer=layer, n_filter=self.hidden_num, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_e')

            ## Encoder
            with tf.variable_scope(name_or_scope='encoder'):
                for idx in range(self.layer_num):
                    with tf.variable_scope(name_or_scope='conv2d_' + str(idx)):
                        layer = _residual_block(inputs=layer, n_filter=self.hidden_num * (idx + 1), name='residual_block_{}'.format(idx))
                        # layer = Conv2d(layer=layer, n_filter=self.hidden_num * (idx + 1), filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_a')
                        # layer = Conv2d(layer=layer, n_filter=self.hidden_num * (idx + 1), filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='conv2d_b')

                        if idx < (self.layer_num - 1):
                            layer = Conv2d(layer=layer, n_filter=self.hidden_num * (idx + 2), filter_size=(3, 3), strides=(2, 2), act=tf.nn.elu, name='conv2d_c')

                layer = ReshapeLayer(prev_layer=layer, shape=(-1, self.s_size[0] * self.s_size[1] * self.hidden_num * self.layer_num), name='e_reshape')

                z = DenseLayer(prev_layer=layer, n_units=self.z_dim, name='z_dense')
                z = tf.nn.tanh(z.outputs, name='z_logit')

            layer = DenseLayer(prev_layer=layer, n_units=self.s_size[0] * self.s_size[1] * self.hidden_num, name='d_input')
            h0_layer = ReshapeLayer(prev_layer=layer, shape=(-1, self.s_size[0], self.s_size[1], self.hidden_num), name='d_reshape')

            ## Decoder
            with tf.variable_scope(name_or_scope='decoder'):
                with tf.variable_scope(name_or_scope='deconv'):
                    for idx in range(self.layer_num):
                        layer = _deconvolution(layer=h0_layer if idx == 0 else layer, idx=idx, name='deconv_' + str(idx))
                    layer.outputs = tf.concat([tf.image.resize_nearest_neighbor(h0_layer.outputs, self.output_size[-1], name='h0_resize'),
                                               layer.outputs], -1)

            with tf.variable_scope('output'):
                layer = Conv2d(layer=layer, n_filter=self.channel, filter_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='d_logit')

        self.variables = tl.layers.get_variables_with_name(name, True, True)

        return layer.outputs, z

class BEGAN:
    def __init__(self, sess, batch_size=16, z_dim=64, img_size=(128, 128), learning_rate=0.0001, channel=3, layer_num=5,
                 hidden_num=30, output_size=[(16, 16), (32, 32), (64, 64), (128, 128)]):
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

    def train(self, z, x):
        self.g.training, self.d.training = True, True
        return self.sess.run([self.d_loss, self.g_loss, self.k_t, self.measure, self.k_update], feed_dict={self.z: z, self.x: x})

    def generate(self, z):
        self.g.training = False
        return self.sess.run(self.g_img, feed_dict={self.z: z})