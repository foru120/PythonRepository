import tensorflow as tf

class Generator:
    def __init__(self, depths=[2048, 1024, 512, 256, 128, 64, 32], s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size
        self.reuse = False

    def __call__(self, inputs, training=False):
        self.training = training

        with tf.variable_scope('g', reuse=self.reuse):
            with tf.variable_scope('reshape'):
                layer = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
                layer = tf.reshape(layer, [-1, self.s_size, self.s_size, self.depths[0]])
                layer = tf.nn.elu(tf.layers.batch_normalization(layer, training=self.training), name='layer')

            for i in range(1, len(self.depths)):
                layer = self.deconvolution(layer, 'deconv'+str(i), i)

            with tf.variable_scope('tanh'):
                layer = tf.tanh(layer, 'layer')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return layer

    def deconvolution(self, layer, name, idx):
        with tf.variable_scope(name):
            layer = tf.layers.conv2d_transpose(layer, self.depths[idx], [5, 5], strides=(2, 2), padding='SAME')
            if idx == 4:
                return layer
            layer = tf.nn.elu(tf.layers.batch_normalization(layer, training=self.training), name='layer')
        return layer

class Discriminator:
    def __init__(self, depths=[16, 32, 64, 128, 256, 512, 1024]):
        self.depths = [3] + depths
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        def convolution(layer, name, idx):
            with tf.variable_scope(name):
                layer = tf.layers.conv2d(layer, self.depths[idx], [5, 5], strides=(2, 2), padding='SAME')
                layer = tf.nn.elu(tf.layers.batch_normalization(layer, training=training), name='layer')
            return layer

        layer = tf.convert_to_tensor(inputs)

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            for i in range(1, len(self.depths)):
                layer = convolution(layer, 'conv' + str(i), i)

            with tf.variable_scope('classify'):
                batch_size = layer.get_shape()[0].value
                reshape = tf.reshape(layer, [batch_size, -1])
                layer = tf.layers.dense(reshape, 2, name='layer')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return layer

class DCGAN:
    def __init__(self, sess, batch_size=100, s_size=4, z_dim=100, img_size=256, learning_rate=0.0002,
                 g_depths=[1024, 512, 256, 128, 64, 32], d_depths=[16, 32, 64, 128, 256, 512]):

        self.sess = sess
        self.batch_size = batch_size
        self.s_size = s_size  # Generator 의 첫 번째 layer
        self.z_dim = z_dim
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.g = Generator(depths=g_depths, s_size=self.s_size)
        self.d = Discriminator(depths=d_depths)
        self._network()

    def _network(self):
        with tf.name_scope('initialize_step'):
            self.randomdata = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim])
            self.traindata = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.img_size, self.img_size, 3])
            self.training = tf.placeholder(dtype=tf.bool, shape=None, name='training')

        with tf.name_scope('dcgan_network'):
            self.generated_img = self.g(self.randomdata, training=self.training)  # batch_size, 256, 256, 3
            g_outputs = self.d(self.generated_img, training=self.training, name='g')
            t_outputs = self.d(self.traindata, training=self.training, name='t')

        self.g_losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self.batch_size], dtype=tf.int64), logits=g_outputs))
        self.d_losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([self.batch_size], dtype=tf.int64), logits=g_outputs)) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([self.batch_size], dtype=tf.int64), logits=t_outputs))

        g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        d_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        self.g_opt_op = g_opt.minimize(self.g_losses, var_list=self.g.variables)
        self.d_opt_op = d_opt.minimize(self.d_losses, var_list=self.d.variables)

        with tf.control_dependencies([self.g_opt_op, self.d_opt_op]):
            self.op = tf.no_op(name='train')

    def train(self, traindata, randomdata):
        return self.sess.run([self.g_losses, self.d_losses, self.op],
                             feed_dict={self.randomdata: randomdata, self.traindata: traindata, self.training: True})

    def generate(self, randomdata):
        return self.sess.run(self.generated_img, feed_dict={self.randomdata: randomdata, self.training: False})

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.randomdata
        images = self.g(inputs, training=True)
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))