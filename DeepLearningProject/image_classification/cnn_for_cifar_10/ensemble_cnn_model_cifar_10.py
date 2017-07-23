import tensorflow as tf

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                self.training = tf.placeholder(tf.bool, name='training')

                self.X = tf.placeholder(tf.float32, [None, 1024], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 32, 32, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10], name='y_data')

            with tf.name_scope('stem_layer') as scope:
                self.W1_sub = tf.get_variable(name='W1_sub', shape=[3, 3, 1, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_sub = tf.nn.conv2d(input=X_img, filter=self.W1_sub, strides=[1, 1, 1, 1], padding='VALID')  # 32x32 -> 30x30
                self.L1_sub = self.batch_norm(input=self.L1_sub, shape=self.L1_sub.get_shape()[-1], training=self.training, convl=True, name='stem_sub1_BN')
                self.L1_sub = self.parametric_relu(self.L1_sub, 'R1_sub')
                self.W2_sub = tf.get_variable(name='W2_sub', shape=[3, 3, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2_sub = tf.nn.conv2d(input=self.L1_sub, filter=self.W2_sub, strides=[1, 1, 1, 1], padding='VALID')  # 30x30 -> 28x28
                self.L2_sub = self.batch_norm(input=self.L2_sub, shape=self.L2_sub.get_shape()[-1], training=self.training, convl=True, name='stem_sub2_BN')
                self.L2_sub = self.parametric_relu(self.L2_sub, 'R2_sub')
                self.W3_sub = tf.get_variable(name='W3_sub', shape=[3, 3, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W3_sub, strides=[1, 1, 1, 1], padding='VALID')  # 28x28 -> 26x26
                self.L3_sub = self.batch_norm(input=self.L3_sub, shape=self.L3_sub.get_shape()[-1], training=self.training, convl=True, name='stem_sub3_BN')
                self.L3_sub = self.parametric_relu(self.L3_sub, 'R3_sub')
                self.L1 = tf.nn.max_pool(value=self.L3_sub, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 26x26 -> 13x13

            with tf.name_scope('inception_layer1') as scope:
                self.L2 = self.inception_model(self.L1, 3, 40, name='inception_layer1')  # 13x13 -> 7x7

            with tf.name_scope('inception_layer2') as scope:
                self.L3 = self.inception_model(self.L2, 3, 80, name='inception_layer2')  # 7x7 -> 4x4

            with tf.name_scope('conv_layer1') as scope:
                self.W4 = tf.get_variable(name='W4', shape=[3, 3, 80, 160], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4 = tf.nn.conv2d(input=self.L3, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME')
                self.L4 = self.batch_norm(self.L4, shape=160, training=self.training, convl=True, name='conv1_BN')
                self.L4 = self.parametric_relu(self.L4, 'R4')
                self.L4 = tf.reshape(self.L4, shape=[-1, 4 * 4 * 160])

            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[4 * 4 * 160, 625], dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b_fc1'))
                self.L_fc1 = tf.matmul(self.L4, self.W_fc1) + self.b_fc1
                self.L_fc1 = self.batch_norm(self.L_fc1, shape=self.L_fc1.get_shape()[-1], training=self.training, convl=False, name='fc1_BN')
                self.L_fc1 = self.parametric_relu(self.L_fc1, 'R_fc1')

            with tf.name_scope('fc_layer2') as scope:
                self.W_fc2 = tf.get_variable(name='W_fc2', shape=[625, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b_fc2'))
                self.L_fc2 = tf.matmul(self.L_fc1, self.W_fc2) + self.b_fc2
                self.L_fc2 = self.batch_norm(self.L_fc2, shape=self.W_fc2.get_shape()[-1], training=self.training, convl=False, name='fc2_BN')
                self.L_fc2 = self.parametric_relu(self.L_fc2, 'R_fc2')

            self.W_out = tf.get_variable(name='W_out', shape=[625, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[10], name='b_out'))
            self.logits = tf.matmul(self.L_fc2, self.W_out) + self.b_out

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + \
                        (0.01 / (2 * tf.to_float(tf.shape(self.Y)[0]))) * tf.reduce_sum(tf.square(self.W_out))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def batch_norm(self, input, shape, training, convl=True, name='BN'):
        beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta')
        scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='scale')
        if convl:
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        else:
            batch_mean, batch_var = tf.nn.moments(input, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)),
                            lambda: (batch_mean, batch_var))
        return tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=0.001, name=name)

    def inception_model(self, x, n, output, name):
        OPL = int(output/4)
        B, H, W, C = x.get_shape()

        with tf.variable_scope(name):
            bias = tf.Variable(tf.constant(value=0.001, shape=[output], name='bias'))

            # 1x1
            W1x1 = tf.get_variable(name='W1x1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L1x1 = tf.nn.conv2d(name='L1x1', input=x, filter=W1x1, strides=[1, 2, 2, 1], padding='SAME')

            # 5x5 -> 1x1, 1x3, 3x1, 1x3, 3x1
            W5x5_sub1 = tf.get_variable(name='W5x5_sub1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub1 = tf.nn.conv2d(name='L5x5_sub1', input=x, filter=W5x5_sub1, strides=[1, 1, 1, 1], padding='SAME')

            W5x5_sub2 = tf.get_variable(name='W5x5_sub2', shape=[1, n, OPL, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub2 = tf.nn.conv2d(name='L5x5_sub2', input=L5x5_sub1, filter=W5x5_sub2, strides=[1, 1, 1, 1], padding='SAME')

            W5x5_sub3 = tf.get_variable(name='W5x5_sub3', shape=[n, 1, OPL, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub3 = tf.nn.conv2d(name='L5x5_sub3', input=L5x5_sub2, filter=W5x5_sub3, strides=[1, 1, 1, 1], padding='SAME')

            W5x5_sub4 = tf.get_variable(name='W5x5_sub4', shape=[1, n, OPL, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub4 = tf.nn.conv2d(name='L5x5_sub4', input=L5x5_sub3, filter=W5x5_sub4, strides=[1, 1, 2, 1], padding='SAME')

            W5x5_sub5 = tf.get_variable(name='W5x5_sub5', shape=[n, 1, OPL, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub5 = tf.nn.conv2d(name='L5x5_sub5', input=L5x5_sub4, filter=W5x5_sub5, strides=[1, 2, 1, 1], padding='SAME')

            # 3x3 -> 1x1, 1x3, 3x1
            W3x3_sub1 = tf.get_variable(name='W3x3_sub1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub1 = tf.nn.conv2d(name='L3x3_sub1', input=x, filter=W3x3_sub1, strides=[1, 1, 1, 1], padding='SAME')

            W3x3_sub2 = tf.get_variable(name='W3x3_sub2', shape=[1, n, OPL, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub2 = tf.nn.conv2d(name='L3x3_sub2', input=L3x3_sub1, filter=W3x3_sub2, strides=[1, 1, 2, 1], padding='SAME')

            W3x3_sub3 = tf.get_variable(name='W3x3_sub3', shape=[n, 1, OPL, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub3 = tf.nn.conv2d(name='L3x3_sub3', input=L3x3_sub2, filter=W3x3_sub3, strides=[1, 2, 1, 1], padding='SAME')

            # max pooling -> max pooling, 1x1
            L_pool = tf.nn.max_pool(name='L_pool', value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            W_pool_sub1 = tf.get_variable(name='W_pool_sub1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L_pool_sub1 = tf.nn.conv2d(name='L_pool_sub1', input=L_pool, filter=W_pool_sub1, strides=[1, 2, 2, 1], padding='SAME')

            tot_layers = tf.concat([L1x1, L5x5_sub5, L3x3_sub3, L_pool_sub1], axis=3)  # Concat in the 4th dim to stack
            tot_layers = self.batch_norm(input=tot_layers, shape=output, training=self.training, convl=True, name='inception_BN')
            tot_layers = self.parametric_relu(tot_layers + bias, 'R_inception')
        return tot_layers