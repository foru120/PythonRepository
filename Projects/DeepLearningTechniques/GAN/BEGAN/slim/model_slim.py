import tensorflow as tf

class Layers:
    def __init__(self):
        self.training = True
        self.output_size = [(16, 16), (32, 32), (64, 64), (128, 128)]
        self.layer_num = 5
        '''Dense Network Parameter'''
        self.growth_rate = 10
        self.dense_block_cnt = 3
        '''Residual Network Parameter'''
        self.default_filter = 50
        self.residual_block_cnt = 4

    def Batch_Norm(self, inputs, act=tf.nn.elu, reuse=False, name='batch_norm'):
        return tf.contrib.layers.batch_norm(inputs=inputs, activation_fn=act, is_training=self.training, updates_collections=None, reuse=reuse, scope=name)

    def Conv2D_Layer(self, inputs, filters, k_size=(3, 3), strides=(2, 2), padding='same', act=tf.nn.elu, reuse=False, name='conv2d'):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=k_size, strides=strides, padding=padding, activation=act, reuse=reuse, name=name)

    def Dense_Layer(self, inputs, filters, act=tf.nn.elu, reuse=False, name='dense'):
        return tf.layers.dense(inputs=inputs, units=filters, activation=act, reuse=reuse, name=name)

    def Dense_Block(self, inputs, name='dense_block'):
        with tf.variable_scope(name):
            '''bottleneck layer (DenseNet-B)'''
            layer = self.Batch_Norm(inputs=inputs, name='bottleneck_batch_norm')
            layer = self.Conv2D_Layer(inputs=layer, filters=4 * self.growth_rate, k_size=(1, 1), strides=(1, 1), act=tf.identity, name='bottleneck_conv')

            '''basic dense layer'''
            layer = self.Batch_Norm(inputs=layer, name='basic_batch_norm')
            layer = self.Conv2D_Layer(inputs=layer, filters=self.growth_rate, k_size=(3, 3), strides=(1, 1), act=tf.identity, name='basic_conv')

            layer = tf.concat([inputs, layer], axis=-1, name='concat')
        return layer

    def Residual_Block(self, inputs, n_filter, name):
        '''
        Improved ResNet
         - original resnet 에서 마지막 단계의 summation 연산 시 ReLU 함수를 사용한 것과 달리 identity 함수를 사용해
           identity mapping 이 이루어지게 된다.
         - full pre-activation: batch normalization 을 residual net 상에서 activation 함수보다 앞으로 오는 구조.
        '''
        with tf.variable_scope(name):
            layer = self.Batch_Norm(inputs=inputs, name='batch_norm_1')
            layer = self.Conv2D_Layer(inputs=layer, filters=n_filter, k_size=(3, 3), strides=(1, 1), act=tf.identity, name='residual_1')
            layer = self.Batch_Norm(inputs=layer, name='batch_norm_2')
            layer = self.Conv2D_Layer(inputs=layer, filters=n_filter, k_size=(3, 3), strides=(1, 1), act=tf.identity, name='residual_2')
            layer = tf.add(inputs, layer)
        return layer

    def Deconv_Layer(self, layer, idx, name='deconv'):
        with tf.variable_scope(name_or_scope=name):
            for i in range(self.dense_block_cnt):
                layer = self.Dense_Block(inputs=layer, name='dense_block_{}'.format(i))
            # for i in range(self.residual_block_cnt):
            #     layer = self.Residual_Block(inputs=layer, n_filter=self.default_filter + (idx * 10), name='residual_block_{}'.format(i))

            if idx == (self.layer_num - 1):
                return layer

            layer = tf.image.resize_nearest_neighbor(layer, self.output_size[idx], name='resize_conv')
        return layer

class Generator(Layers):
    def __init__(self, hidden_num, s_size=(4, 4), channel=3):
        super(Generator, self).__init__()
        self.hidden_num = hidden_num
        self.s_size = s_size
        self.channel = channel

    def __call__(self, inputs, reuse=False, name='Generator'):
        self.reuse = reuse

        with tf.variable_scope(name_or_scope=name, reuse=self.reuse):
            with tf.variable_scope(name_or_scope='input'):
                layer = self.Dense_Layer(inputs=inputs, filters=self.hidden_num * self.s_size[0] * self.s_size[1], act=tf.identity, name='dense')
                h0_layer = tf.reshape(layer, shape=(-1, self.s_size[0], self.s_size[1], self.hidden_num), name='reshape')

            with tf.variable_scope(name_or_scope='deconv'):
                for idx in range(self.layer_num):
                    layer = self.Deconv_Layer(layer=h0_layer if idx == 0 else layer, idx=idx, name='deconv_'+str(idx))
                layer = tf.concat([tf.image.resize_nearest_neighbor(h0_layer, self.output_size[-1], name='h0_resize'), layer], -1)

            with tf.variable_scope('output'):
                layer = self.Conv2D_Layer(inputs=layer, filters=self.channel, k_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='g_logit')

        self.variables = [var for var in tf.trainable_variables() if name in var.name]

        return layer

class Discriminator(Layers):
    def __init__(self, hidden_num, s_size=(4, 4), channel=3, z_dim=100):
        super(Discriminator, self).__init__()
        self.hidden_num = hidden_num
        self.s_size = s_size
        self.channel = channel
        self.z_dim = z_dim

    def __call__(self, inputs, reuse=False, name='Discriminator'):
        self.reuse = reuse

        with tf.variable_scope(name_or_scope=name, reuse=self.reuse):
            ## Encoder
            with tf.variable_scope(name_or_scope='encoder'):
                for idx in range(self.layer_num):
                    with tf.variable_scope(name_or_scope='conv2d_' + str(idx)):
                        layer = self.Conv2D_Layer(inputs=inputs if idx == 0 else layer, filters=self.hidden_num * (idx + 1), k_size=(3, 3), strides=(1, 1), name='conv2d_a')
                        layer = self.Conv2D_Layer(inputs=layer, filters=self.hidden_num * (idx + 1), k_size=(3, 3), strides=(1, 1), name='conv2d_b')

                        if idx < (self.layer_num - 1):
                            layer = self.Conv2D_Layer(inputs=layer, filters=self.hidden_num * (idx + 2), k_size=(3, 3), strides=(2, 2), name='conv2d_c')

                layer = tf.reshape(layer, shape=(-1, self.s_size[0] * self.s_size[1] * self.hidden_num * self.layer_num), name='e_reshape')

                z = self.Dense_Layer(inputs=layer, filters=self.z_dim, act=tf.identity, name='z_dense')
                z = tf.nn.tanh(z, name='z_logit')

            ## Decoder
            with tf.variable_scope(name_or_scope='decoder'):
                layer = self.Dense_Layer(inputs=layer, filters=self.s_size[0] * self.s_size[1] * self.hidden_num, name='d_input')
                h0_layer = tf.reshape(layer, shape=(-1, self.s_size[0], self.s_size[1], self.hidden_num), name='d_reshape')

                with tf.variable_scope(name_or_scope='deconv'):
                    for idx in range(self.layer_num):
                        layer = self.Deconv_Layer(layer=h0_layer if idx == 0 else layer, idx=idx, name='deconv_'+str(idx))
                    layer = tf.concat([tf.image.resize_nearest_neighbor(h0_layer, self.output_size[-1], name='h0_resize'), layer], -1)

            with tf.variable_scope('output'):
                layer = self.Conv2D_Layer(inputs=layer, filters=self.channel, k_size=(3, 3), strides=(1, 1), act=tf.nn.elu, name='d_logit')

        self.variables = [var for var in tf.trainable_variables() if name in var.name]

        return layer, z

class BEGAN:
    def __init__(self, sess, batch_size=16, z_dim=64, img_size=(128, 128), learning_rate=0.0001, channel=3, layer_num=5,
                 hidden_num=50):
        self.sess = sess
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.img_size = img_size
        self.channel = channel
        self.layer_num = layer_num
        self.learning_rate = learning_rate
        self.gamma = 0.5
        self.lambda_k = 0.001

        self.g = Generator(hidden_num=hidden_num, s_size=(8, 8), channel=self.channel)
        self.d = Discriminator(hidden_num=hidden_num, s_size=(8, 8), channel=self.channel, z_dim=z_dim)
        self._network()

    def _network(self):
        with tf.variable_scope('input_data'):
            self.z = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.z_dim), name='z_data')
            self.x = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size[0], self.img_size[1], self.channel), name='x_data')
            self.k_t = tf.Variable(0., trainable=False, dtype=tf.float32, name='k_t')

        with tf.variable_scope('network'):
            self.g_digit = self.g(inputs=self.z, reuse=False, name='Generator')
            self.AE_G, self.G_d_z = self.d(inputs=self.g_digit, reuse=False, name='Discriminator')
            self.AE_X, self.X_d_z = self.d(inputs=self.x, reuse=True, name='Discriminator')
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
        Layers.training = True
        return self.sess.run([self.d_loss, self.g_loss, self.k_t, self.measure, self.k_update], feed_dict={self.z: z, self.x: x})

    def generate(self, z):
        Layers.training = False
        return self.sess.run(self.g_img, feed_dict={self.z: z})