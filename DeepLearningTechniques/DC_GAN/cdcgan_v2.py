import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

class Generator:
    '''
        1. output layer 를 제외한 모든 구간에서 batch normalization 사용
        2. output layer 를 제외한 모든 layer 의 activation function 은 ReLU 사용
        3. output layer 의 입력값에 gaussian noise 추가
    '''
    def __init__(self, depths, s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size
        self.output_size = [(3, 4), (6, 8), (12, 16), (24, 32), (48, 64), (96, 128), (192, 256)]
        self.w_init = tf.random_normal_initializer(stddev=0.02)
        self.gamma_init = tf.random_normal_initializer(1., 0.02)
        self.reuse = None

    def __call__(self, inputs, training=False, reuse=False, name='g'):
        def batch_norm(layer, name):
            if training:
                with tf.variable_scope(name_or_scope=name):
                    return BatchNormLayer(layer, act=tf.nn.relu, is_train=True, gamma_init=self.gamma_init)
            else:
                with tf.variable_scope(name_or_scope=name, reuse=True):
                    return BatchNormLayer(layer, act=tf.nn.relu, is_train=False, gamma_init=self.gamma_init)

        def deconvolution(layer, name, idx):
            with tf.variable_scope(name):
                layer = tl.layers.PadLayer(layer, padding=[[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
                if idx == 6:
                    layer = Conv2d(layer, n_filter=self.depths[idx], filter_size=(5, 5), strides=(1, 1), act=tf.identity, padding='SAME',
                                   W_init=self.w_init, name='conv2d')
                else:
                    layer = Conv2d(layer, n_filter=self.depths[idx], filter_size=(5, 5), strides=(1, 1), act=tf.identity, padding='SAME',
                                   W_init=self.w_init, name='conv2d')
                    layer = batch_norm(layer, 'batchnorm')
            return layer

        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with tf.variable_scope('input'):
                layer = InputLayer(inputs, name='input')
                layer = DenseLayer(layer, n_units=self.depths[0] * self.output_size[0][0] * self.output_size[0][1],
                                   W_init=self.w_init, act=tf.identity, name='dense')
                layer = ReshapeLayer(layer, shape=[-1, self.output_size[0][0], self.output_size[0][1], self.depths[0]], name='reshape')
                layer = batch_norm(layer, 'batch_norm')

            for i in range(1, len(self.depths)):
                layer = deconvolution(layer, 'deconv'+str(i), i)

            with tf.variable_scope('tanh'):
                layer = tf.nn.tanh(layer.outputs, 'g_output')

        self.variables = tl.layers.get_variables_with_name(name, True, True)

        return layer

class Discriminator:
    '''
        1. input layer 를 제외한 모든 구간에서 batch normalization 사용
        2. 모든 layer 의 activation function 은 LeakyReLU 사용
    '''
    def __init__(self, depths):
        self.depths = [3] + depths
        self.w_init = tf.random_normal_initializer(stddev=0.02)
        self.gamma_init = tf.random_normal_initializer(1., 0.02)

    def __call__(self, inputs, training=False, reuse=False):
        def batch_norm(layer, name):
            if training:
                with tf.variable_scope(name_or_scope=name):
                    layer = BatchNormLayer(layer, act=lambda x : tl.act.lrelu(x, 0.2), is_train=True, gamma_init=self.gamma_init)
                    return layer
            else:
                with tf.variable_scope(name_or_scope=name, reuse=True):
                    layer = BatchNormLayer(layer, act=lambda x : tl.act.lrelu(x, 0.2), is_train=False, gamma_init=self.gamma_init)
                    return layer

        def convolution(layer, name, idx):
            with tf.variable_scope(name):
                layer = Conv2d(layer, n_filter=self.depths[idx], filter_size=(5, 5), strides=(2, 2), padding='SAME', act=None, W_init=self.w_init, name='conv2d')
                layer = batch_norm(layer, 'batch_norm')
            return layer

        with tf.variable_scope('d', reuse=reuse):
            layer = InputLayer(inputs)
            layer = Conv2d(layer, n_filter=self.depths[0], filter_size=(5, 5), strides=(2, 2), padding='SAME',
                           act=lambda x : tl.act.lrelu(x, 0.2), W_init=self.w_init, name='conv2d')

            for i in range(1, len(self.depths)):
                layer = convolution(layer, 'conv' + str(i), i)

            with tf.variable_scope('classify'):
                batch_size = layer.outputs.get_shape()[0].value
                layer = ReshapeLayer(layer, shape=[batch_size, -1])
                layer = DenseLayer(layer, n_units=2, W_init=self.w_init, act=tf.identity, name='d_output')

        self.variables = tl.layers.get_variables_with_name('d', True, True)

        return layer.outputs

class CDCGAN:
    def __init__(self, sess, batch_size=100, z_dim=100, img_size=(192, 256), learning_rate=0.0002,
                 g_depths=[1024, 512, 256, 128, 64, 32], d_depths=[16, 32, 64, 128, 256, 512]):
        self.sess = sess
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.g1 = Generator(depths=g_depths)
        self.g2 = Generator(depths=g_depths)
        self.d = Discriminator(depths=d_depths)
        self.reuse = False
        self.training = True
        self._network()

    def _network(self):
        with tf.name_scope('initialize_step'):
            self.batch_z = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim], name='batch_z')
            self.train_x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.img_size[0], self.img_size[1], 3], name='train_x')

        with tf.name_scope('dcgan_network'):
            self.g1_logits = self.g1(self.batch_z, training=self.training, reuse=self.reuse, name='g1')
            self.g2_logits = self.g2(self.batch_z, training=self.training, reuse=self.reuse, name='g2')
            self.d1_logits = self.d(self.g1_logits, training=self.training, reuse=False)
            self.d2_logits = self.d(self.g2_logits, training=self.training, reuse=True)
            self.d3_logits = self.d(self.train_x, training=self.training, reuse=True)

        self.g1_losses = tl.cost.sigmoid_cross_entropy(output=self.d1_logits, target=tf.ones_like(self.d1_logits), name='g1_losses')
        self.d1_losses = tl.cost.sigmoid_cross_entropy(output=self.d1_logits, target=tf.zeros_like(self.d1_logits), name='d1_losses')
        self.g2_losses = tl.cost.sigmoid_cross_entropy(output=self.d2_logits, target=tf.ones_like(self.d2_logits), name='g2_losses')
        self.d2_losses = tl.cost.sigmoid_cross_entropy(output=self.d2_logits, target=tf.zeros_like(self.d2_logits), name='d2_losses')
        self.d3_losses = tl.cost.sigmoid_cross_entropy(output=self.d3_logits, target=tf.ones_like(self.d3_logits), name='d3_losses')
        self.d_losses = self.d1_losses + self.d2_losses + self.d3_losses

        self.tot_g1_losses = tf.where(tf.less_equal(self.g1_losses, self.g2_losses), self.g1_losses, self.g1_losses + self.g2_losses/2)
        self.tot_g2_losses = tf.where(tf.less_equal(self.g2_losses, self.g1_losses), self.g2_losses, self.g2_losses + self.g1_losses/2)

        self.g1_opt_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.tot_g1_losses, var_list=self.g1.variables)
        self.g2_opt_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.tot_g2_losses, var_list=self.g2.variables)
        self.d_opt_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.d_losses, var_list=self.d.variables)

    def g_train(self, batch_z):
        self.reuse = False
        self.training = True
        return self.sess.run([self.g1_losses, self.g2_losses, self.tot_g1_losses, self.tot_g2_losses, self.g1_opt_op, self.g2_opt_op], feed_dict={self.batch_z: batch_z})

    def d_train(self, batch_z, train_x):
        self.training = True
        return self.sess.run([self.d_losses, self.d_opt_op], feed_dict={self.batch_z: batch_z, self.train_x: train_x})

    def generate(self, batch_z):
        self.reuse = True
        self.training = False
        return self.sess.run([self.g1_logits, self.g2_logits], feed_dict={self.batch_z: batch_z})