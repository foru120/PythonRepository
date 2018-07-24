import tensorflow as tf

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')

                self.X = tf.placeholder(tf.float32, [None, 1024], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 32, 32, 1])
                self.Y = tf.placeholder(tf.float32, [None, 10], name='y_data')

            with tf.name_scope('stem_layer') as scope:
                self.W1_sub = tf.get_variable(name='W1_sub', shape=[1, 3, 1, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_sub = tf.nn.conv2d(input=X_img, filter=self.W1_sub, strides=[1, 1, 1, 1], padding='VALID')  # 32x32 -> 32x30
                self.L1_sub = self.BN(input=self.L1_sub, training=self.training, name='L1_sub_BN')
                self.L1_sub = self.parametric_relu(self.L1_sub, 'R1_sub')
                self.W2_sub = tf.get_variable(name='W2_sub', shape=[3, 1, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2_sub = tf.nn.conv2d(input=self.L1_sub, filter=self.W2_sub, strides=[1, 1, 1, 1], padding='VALID')  # 32x30 -> 30x30
                self.L2_sub = self.BN(input=self.L2_sub, training=self.training, name='L2_sub_BN')
                self.L2_sub = self.parametric_relu(self.L2_sub, 'R2_sub')
                self.W3_sub = tf.get_variable(name='W3_sub', shape=[1, 3, 20, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_sub = tf.nn.conv2d(input=self.L2_sub, filter=self.W3_sub, strides=[1, 1, 1, 1], padding='VALID')  # 30x30 -> 30x28
                self.L3_sub = self.BN(input=self.L3_sub, training=self.training, name='L3_sub_BN')
                self.L3_sub = self.parametric_relu(self.L3_sub, 'R3_sub')
                self.W4_sub = tf.get_variable(name='W4_sub', shape=[3, 1, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4_sub = tf.nn.conv2d(input=self.L3_sub, filter=self.W4_sub, strides=[1, 1, 1, 1], padding='VALID')  # 30x28 -> 28x28
                self.L4_sub = self.BN(input=self.L4_sub, training=self.training, name='L4_sub_BN')
                self.L4_sub = self.parametric_relu(self.L4_sub, 'R4_sub')
                self.W5_sub = tf.get_variable(name='W5_sub', shape=[1, 3, 40, 80], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L5_sub = tf.nn.conv2d(input=self.L4_sub, filter=self.W5_sub, strides=[1, 1, 1, 1], padding='VALID')  # 28x28 -> 28x26
                self.L5_sub = self.BN(input=self.L5_sub, training=self.training, name='L5_sub_BN')
                self.L5_sub = self.parametric_relu(self.L5_sub, 'R5_sub')
                self.W6_sub = tf.get_variable(name='W6_sub', shape=[3, 1, 80, 80], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L6_sub = tf.nn.conv2d(input=self.L5_sub, filter=self.W6_sub, strides=[1, 1, 1, 1], padding='VALID')  # 28x26 -> 26x26
                self.L6_sub = self.BN(input=self.L6_sub, training=self.training, name='L6_sub_BN')
                self.L6_sub = self.parametric_relu(self.L6_sub, 'R6_sub')
                # self.L1 = tf.nn.max_pool(value=self.L3_sub, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 26x26 -> 13x13

            with tf.name_scope('inception_layer1') as scope:
                self.Inception_L1 = self.inception_A(self.L6_sub, 3, 120, name='inception_layer1')  # 26x26 -> 13x13

            with tf.name_scope('inception_layer2') as scope:
                self.Inception_L2 = self.inception_B(self.Inception_L1, 3, 160, name='inception_layer2')  # 13x13 -> 7x7

            with tf.name_scope('inception_layer3') as scope:
                self.Inception_L3 = self.inception_C(self.Inception_L2, 3, 200, name='inception_layer3')  # 7x7 -> 4x4
                self.Inception_L3 = tf.layers.dropout(inputs=self.Inception_L3, rate=self.dropout_rate, training=self.training)
                # self.Inception_L3 = tf.reshape(self.Inception_L3, shape=[-1, 4 * 4 * 200])

            with tf.name_scope('conv_layer1') as scope:
                self.W4 = tf.get_variable(name='W4', shape=[3, 3, 200, 240], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4 = tf.nn.conv2d(input=self.Inception_L3, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME')
                self.L4 = self.BN(input=self.L4, training=self.training, name='conv1_BN')
                self.L4 = self.parametric_relu(self.L4, 'R4')
                # self.L4 = tf.reshape(self.L4, shape=[-1, 4 * 4 * 240])
                self.L4 = tf.layers.dropout(inputs=self.L4, rate=self.dropout_rate, training=self.training)

            with tf.name_scope('conv_layer2') as scope:
                self.W5 = tf.get_variable(name='W5', shape=[3, 3, 240, 300], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L5 = tf.nn.conv2d(input=self.L4, filter=self.W5, strides=[1, 1, 1, 1], padding='SAME')
                self.L5 = self.BN(input=self.L5, training=self.training, name='conv2_BN')
                self.L5 = self.parametric_relu(self.L5, 'R5')
                self.L5 = tf.reshape(self.L5, shape=[-1, 4 * 4 * 300])
                self.L5 = tf.layers.dropout(inputs=self.L5, rate=self.dropout_rate, training=self.training)

            with tf.name_scope('fc_layer1') as scope:
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[4 * 4 * 300, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc1'))
                self.L_fc1 = tf.matmul(self.L5, self.W_fc1) + self.b_fc1
                self.L_fc1 = self.BN(input=self.L_fc1, training=self.training, name='fc1_BN')
                self.L_fc1 = self.parametric_relu(self.L_fc1, 'R_fc1')
                self.L_fc1 = tf.layers.dropout(inputs=self.L_fc1, rate=self.dropout_rate, training=self.training)

            with tf.name_scope('fc_layer2') as scope:
                self.W_fc2 = tf.get_variable(name='W_fc2', shape=[1000, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc2 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc2'))
                self.L_fc2 = tf.matmul(self.L_fc1, self.W_fc2) + self.b_fc2
                self.L_fc2 = self.BN(input=self.L_fc2, training=self.training, name='fc2_BN')
                self.L_fc2 = self.parametric_relu(self.L_fc2, 'R_fc2')
                self.L_fc2 = tf.layers.dropout(inputs=self.L_fc2, rate=self.dropout_rate, training=self.training)

            self.W_out = tf.get_variable(name='W_out', shape=[1000, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[10], name='b_out'))
            self.logits = tf.matmul(self.L_fc2, self.W_out) + self.b_out

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + \
                        (0.01 / (2 * tf.to_float(tf.shape(self.Y)[0]))) * tf.reduce_sum(tf.square(self.W_out))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.cost)
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

    def BN(self, input, training, name):
        # if training is True:
        #     bn = tf.contrib.layers.batch_norm(input, decay, scale=scale, is_training=True, updates_collections=None, scope=name)
        # else:
        #     bn = tf.contrib.layers.batch_norm(input, decay, scale=scale, is_training=True, updates_collections=None, scope=name)
        return tf.contrib.layers.batch_norm(input, decay=0.9, scale=False, is_training=training, updates_collections=None, scope=name, zero_debias_moving_mean=True)

    def inception_A(self, x, n, output, name):
        OPL = int(output/4)
        B, H, W, C = x.get_shape()

        with tf.variable_scope(name):
            # 1x1
            W1x1 = tf.get_variable(name='W1x1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L1x1 = tf.nn.conv2d(name='L1x1', input=x, filter=W1x1, strides=[1, 2, 2, 1], padding='SAME')
            L1x1 = self.BN(input=L1x1, training=self.training, name='inceptionA_L1x1_BN')
            L1x1 = self.parametric_relu(L1x1, 'inceptionA_L1x1_R')

            # 5x5 -> 1x1, 3x3, 3x3
            W5x5_sub1 = tf.get_variable(name='W5x5_sub1', shape=[1, 1, C, 15], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub1 = tf.nn.conv2d(name='L5x5_sub1', input=x, filter=W5x5_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub1 = self.BN(input=L5x5_sub1, training=self.training, name='inceptionA_L5x5_sub1_BN')
            L5x5_sub1 = self.parametric_relu(L5x5_sub1, 'inceptionA_L5x5_sub1_R')

            W5x5_sub2 = tf.get_variable(name='W5x5_sub2', shape=[n, n, 15, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub2 = tf.nn.conv2d(name='L5x5_sub2', input=L5x5_sub1, filter=W5x5_sub2, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub2 = self.BN(input=L5x5_sub2, training=self.training, name='inceptionA_L5x5_sub2_BN')
            L5x5_sub2 = self.parametric_relu(L5x5_sub2, 'inceptionA_L5x5_sub2_R')

            W5x5_sub3 = tf.get_variable(name='W5x5_sub3', shape=[n, n, OPL, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub3 = tf.nn.conv2d(name='L5x5_sub3', input=L5x5_sub2, filter=W5x5_sub3, strides=[1, 2, 2, 1], padding='SAME')
            L5x5_sub3 = self.BN(input=L5x5_sub3, training=self.training, name='inceptionA_L5x5_sub3_BN')
            L5x5_sub3 = self.parametric_relu(L5x5_sub3, 'inceptionA_L5x5_sub3_R')

            # 3x3 -> 1x1, 3x3
            W3x3_sub1 = tf.get_variable(name='W3x3_sub1', shape=[1, 1, C, 15], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub1 = tf.nn.conv2d(name='L3x3_sub1', input=x, filter=W3x3_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L3x3_sub1 = self.BN(input=L3x3_sub1, training=self.training, name='inceptionA_L3x3_sub1_BN')
            L3x3_sub1 = self.parametric_relu(L3x3_sub1, 'inceptionA_L3x3_sub1_R')

            W3x3_sub2 = tf.get_variable(name='W3x3_sub2', shape=[n, n, 15, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub2 = tf.nn.conv2d(name='L3x3_sub2', input=L3x3_sub1, filter=W3x3_sub2, strides=[1, 2, 2, 1], padding='SAME')
            L3x3_sub2 = self.BN(input=L3x3_sub2, training=self.training, name='inceptionA_L3x3_sub2_BN')
            L3x3_sub2 = self.parametric_relu(L3x3_sub2, 'inceptionA_L3x3_sub2_R')

            # avg pooling -> avg pooling, 1x1
            L_pool = tf.nn.avg_pool(name='L_pool', value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            W_pool_sub1 = tf.get_variable(name='W_pool_sub1', shape=[1, 1, C, OPL], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L_pool_sub1 = tf.nn.conv2d(name='L_pool_sub1', input=L_pool, filter=W_pool_sub1, strides=[1, 2, 2, 1], padding='SAME')
            L_pool_sub1 = self.BN(input=L_pool_sub1, training=self.training, name='inceptionA_L_pool_sub1_BN')
            L_pool_sub1 = self.parametric_relu(L_pool_sub1, 'inceptionA_L_pool_sub1_R')

            tot_layers = tf.concat([L1x1, L5x5_sub3, L3x3_sub2, L_pool_sub1], axis=3)  # Concat in the 4th dim to stack
        return tot_layers

    def inception_B(self, x, n, output, name):
        OPL = int(output/4)
        B, H, W, C = x.get_shape()

        with tf.variable_scope(name):
            # 1x1
            W1x1 = tf.get_variable(name='W1x1', shape=[1, 1, C, 60], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L1x1 = tf.nn.conv2d(name='L1x1', input=x, filter=W1x1, strides=[1, 2, 2, 1], padding='SAME')
            L1x1 = self.BN(input=L1x1, training=self.training, name='inceptionB_L1x1_BN')
            L1x1 = self.parametric_relu(L1x1, 'inceptionB_L1x1_R')

            # 5x5 -> 1x1, 1x3, 3x1, 1x3, 3x1
            W5x5_sub1 = tf.get_variable(name='W5x5_sub1', shape=[1, 1, C, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub1 = tf.nn.conv2d(name='L5x5_sub1', input=x, filter=W5x5_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub1 = self.BN(input=L5x5_sub1, training=self.training, name='inceptionB_L5x5_sub1_BN')
            L5x5_sub1 = self.parametric_relu(L5x5_sub1, 'inceptionB_L5x5_sub1_R')

            W5x5_sub2 = tf.get_variable(name='W5x5_sub2', shape=[1, n, 20, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub2 = tf.nn.conv2d(name='L5x5_sub2', input=L5x5_sub1, filter=W5x5_sub2, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub2 = self.BN(input=L5x5_sub2, training=self.training, name='inceptionB_L5x5_sub2_BN')
            L5x5_sub2 = self.parametric_relu(L5x5_sub2, 'inceptionB_L5x5_sub2_R')

            W5x5_sub3 = tf.get_variable(name='W5x5_sub3', shape=[n, 1, 20, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub3 = tf.nn.conv2d(name='L5x5_sub3', input=L5x5_sub2, filter=W5x5_sub3, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub3 = self.BN(input=L5x5_sub3, training=self.training, name='inceptionB_L5x5_sub3_BN')
            L5x5_sub3 = self.parametric_relu(L5x5_sub3, 'inceptionB_L5x5_sub3_R')

            W5x5_sub4 = tf.get_variable(name='W5x5_sub4', shape=[1, n, 30, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub4 = tf.nn.conv2d(name='L5x5_sub4', input=L5x5_sub3, filter=W5x5_sub4, strides=[1, 1, 2, 1], padding='SAME')
            L5x5_sub4 = self.BN(input=L5x5_sub4, training=self.training, name='inceptionB_L5x5_sub4_BN')
            L5x5_sub4 = self.parametric_relu(L5x5_sub4, 'inceptionB_L5x5_sub4_R')

            W5x5_sub5 = tf.get_variable(name='W5x5_sub5', shape=[n, 1, 30, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub5 = tf.nn.conv2d(name='L5x5_sub5', input=L5x5_sub4, filter=W5x5_sub5, strides=[1, 2, 1, 1], padding='SAME')
            L5x5_sub5 = self.BN(input=L5x5_sub5, training=self.training, name='inceptionB_L5x5_sub5_BN')
            L5x5_sub5 = self.parametric_relu(L5x5_sub5, 'inceptionB_L5x5_sub5_R')

            # 3x3 -> 1x1, 1x3, 3x1
            W3x3_sub1 = tf.get_variable(name='W3x3_sub1', shape=[1, 1, C, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub1 = tf.nn.conv2d(name='L3x3_sub1', input=x, filter=W3x3_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L3x3_sub1 = self.BN(input=L3x3_sub1, training=self.training, name='inceptionB_L3x3_sub1_BN')
            L3x3_sub1 = self.parametric_relu(L3x3_sub1, 'inceptionB_L3x3_sub1_R')

            W3x3_sub2 = tf.get_variable(name='W3x3_sub2', shape=[1, n, 20, 30], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub2 = tf.nn.conv2d(name='L3x3_sub2', input=L3x3_sub1, filter=W3x3_sub2, strides=[1, 1, 2, 1], padding='SAME')
            L3x3_sub2 = self.BN(input=L3x3_sub2, training=self.training, name='inceptionB_L3x3_sub2_BN')
            L3x3_sub2 = self.parametric_relu(L3x3_sub2, 'inceptionB_L3x3_sub2_R')

            W3x3_sub3 = tf.get_variable(name='W3x3_sub3', shape=[n, 1, 30, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub3 = tf.nn.conv2d(name='L3x3_sub3', input=L3x3_sub2, filter=W3x3_sub3, strides=[1, 2, 1, 1], padding='SAME')
            L3x3_sub3 = self.BN(input=L3x3_sub3, training=self.training, name='inceptionB_L3x3_sub3_BN')
            L3x3_sub3 = self.parametric_relu(L3x3_sub3, 'inceptionB_L3x3_sub3_R')

            # max pooling -> max pooling, 1x1
            L_pool = tf.nn.avg_pool(name='L_pool', value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            W_pool_sub1 = tf.get_variable(name='W_pool_sub1', shape=[1, 1, C, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L_pool_sub1 = tf.nn.conv2d(name='L_pool_sub1', input=L_pool, filter=W_pool_sub1, strides=[1, 2, 2, 1], padding='SAME')
            L_pool_sub1 = self.BN(input=L_pool_sub1, training=self.training, name='inceptionB_L_pool_sub1_BN')
            L_pool_sub1 = self.parametric_relu(L_pool_sub1, 'inceptionB_L_pool_sub1_R')

            tot_layers = tf.concat([L1x1, L5x5_sub5, L3x3_sub3, L_pool_sub1], axis=3)  # Concat in the 4th dim to stack
        return tot_layers

    def inception_C(self, x, n, output, name):
        OPL = int(output/4)
        B, H, W, C = x.get_shape()

        with tf.variable_scope(name):
            # 1x1
            W1x1 = tf.get_variable(name='W1x1', shape=[1, 1, C, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L1x1 = tf.nn.conv2d(name='L1x1', input=x, filter=W1x1, strides=[1, 2, 2, 1], padding='SAME')
            L1x1 = self.BN(input=L1x1, training=self.training, name='inceptionC_L1x1_BN')
            L1x1 = self.parametric_relu(L1x1, 'inceptionC_L1x1_R')

            # 5x5 -> 1x1, 1x3, 3x1, 1x3, 3x1
            W5x5_sub1 = tf.get_variable(name='W5x5_sub1', shape=[1, 1, C, 100], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub1 = tf.nn.conv2d(name='L5x5_sub1', input=x, filter=W5x5_sub1, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub1 = self.BN(input=L5x5_sub1, training=self.training, name='inceptionC_L5x5_sub1_BN')
            L5x5_sub1 = self.parametric_relu(L5x5_sub1, 'inceptionC_L5x5_sub1_R')

            W5x5_sub2 = tf.get_variable(name='W5x5_sub2', shape=[n, n, 100, 150], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub2 = tf.nn.conv2d(name='L5x5_sub2', input=L5x5_sub1, filter=W5x5_sub2, strides=[1, 2, 2, 1], padding='SAME')
            L5x5_sub2 = self.BN(input=L5x5_sub2, training=self.training, name='inceptionC_L5x5_sub2_BN')
            L5x5_sub2 = self.parametric_relu(L5x5_sub2, 'inceptionC_L5x5_sub2_R')

            W5x5_sub3 = tf.get_variable(name='W5x5_sub3', shape=[n, 1, 150, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub3 = tf.nn.conv2d(name='L5x5_sub3', input=L5x5_sub2, filter=W5x5_sub3, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub3 = self.BN(input=L5x5_sub3, training=self.training, name='inceptionC_L5x5_sub3_BN')
            L5x5_sub3 = self.parametric_relu(L5x5_sub3, 'inceptionC_L5x5_sub3_R')

            W5x5_sub4 = tf.get_variable(name='W5x5_sub4', shape=[1, n, 150, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L5x5_sub4 = tf.nn.conv2d(name='L5x5_sub4', input=L5x5_sub2, filter=W5x5_sub4, strides=[1, 1, 1, 1], padding='SAME')
            L5x5_sub4 = self.BN(input=L5x5_sub4, training=self.training, name='inceptionC_L5x5_sub4_BN')
            L5x5_sub4 = self.parametric_relu(L5x5_sub4, 'inceptionC_L5x5_sub4_R')

            # 3x3 -> 1x1, 1x3, 3x1
            W3x3_sub1 = tf.get_variable(name='W3x3_sub1', shape=[1, 1, C, 100], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub1 = tf.nn.conv2d(name='L3x3_sub1', input=x, filter=W3x3_sub1, strides=[1, 2, 2, 1], padding='SAME')
            L3x3_sub1 = self.BN(input=L3x3_sub1, training=self.training, name='inceptionC_L3x3_sub1_BN')
            L3x3_sub1 = self.parametric_relu(L3x3_sub1, 'inceptionC_L3x3_sub1_R')

            W3x3_sub2 = tf.get_variable(name='W3x3_sub2', shape=[1, n, 100, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub2 = tf.nn.conv2d(name='L3x3_sub2', input=L3x3_sub1, filter=W3x3_sub2, strides=[1, 1, 1, 1], padding='SAME')
            L3x3_sub2 = self.BN(input=L3x3_sub2, training=self.training, name='inceptionC_L3x3_sub2_BN')
            L3x3_sub2 = self.parametric_relu(L3x3_sub2, 'inceptionC_L3x3_sub2_R')

            W3x3_sub3 = tf.get_variable(name='W3x3_sub3', shape=[n, 1, 100, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L3x3_sub3 = tf.nn.conv2d(name='L3x3_sub3', input=L3x3_sub1, filter=W3x3_sub3, strides=[1, 1, 1, 1], padding='SAME')
            L3x3_sub3 = self.BN(input=L3x3_sub3, training=self.training, name='inceptionC_L3x3_sub3_BN')
            L3x3_sub3 = self.parametric_relu(L3x3_sub3, 'inceptionB_L3x3_sub1_R')

            # max pooling -> max pooling, 1x1
            L_pool = tf.nn.avg_pool(name='L_pool', value=x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            W_pool_sub1 = tf.get_variable(name='W_pool_sub1', shape=[1, 1, C, 20], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            L_pool_sub1 = tf.nn.conv2d(name='L_pool_sub1', input=L_pool, filter=W_pool_sub1, strides=[1, 2, 2, 1], padding='SAME')
            L_pool_sub1 = self.BN(input=L_pool_sub1, training=self.training, name='inceptionC_L_pool_sub1_BN')
            L_pool_sub1 = self.parametric_relu(L_pool_sub1, 'inceptionC_L_pool_sub1_R')

            tot_layers = tf.concat([L1x1, L5x5_sub3, L5x5_sub4, L3x3_sub2, L3x3_sub3, L_pool_sub1], axis=3)  # Concat in the 4th dim to stack
        return tot_layers