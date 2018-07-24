import tensorflow as tf

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.class_num = 10
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer'):
                self.dropout_rate = tf.Variable(tf.constant(value=0.5), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')
                self.X = tf.placeholder(tf.float32, [None, 128*128], name='x_data')
                X_img = tf.reshape(self.X, shape=[-1, 128, 128, 1])
                self.Y = tf.placeholder(tf.float32, [None, self.class_num], name='y_data')

            with tf.name_scope('conv1'):
                ####################################################################################################
                ## 논문에서는 Kernel Size 7x7, Stride는 2로 되어있으나 ImageNet의 이미지 크기와 다르기 때문에 5x5, S=1 로 진행함
                ## Output: 128x128
                ####################################################################################################
                self.W1_sub = tf.get_variable(name='W1_sub', shape=[5,5,1,64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_sub = tf.nn.conv2d(input=X_img, filter=self.W1_sub, strides=[1,1,1,1], padding='SAME')
                self.L1_sub = self.parametric_relu(self.L1_sub, 'R_conv1_1')

            with tf.name_scope('conv2_x'):
                self.W2_sub = tf.get_variable(name='W2_sub', shape=[3,3,64,64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # Pooling /2
                self.L2_sub = tf.nn.max_pool(value=self.L1_sub, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
                ####################################################################################################
                ## 2N개 Layer  : (1 Layer + 1 Shortcut Connection layer) x N개
                ## Shortcut Connection에서 차원을 늘려야 하는 경우 Projection 적용, 아닌 경우 1개 Layer를 생략하여 Shorcut Connection 구현
                ## Projection 시 Linear 하도록 활성화 함수를 쓰지 않는다.
                ## Output: 64x64
                ####################################################################################################
                # 2-1
                self.L2_sub_1 = tf.nn.conv2d(input=self.L2_sub, filter=self.W2_sub, strides=[1,1,1,1], padding='SAME')
                self.L2_sub_1 = self.BN(input=self.L2_sub_1, scale=True, training=self.training, name='Conv2_sub_BN_1')
                self.L2_sub_1_r = self.parametric_relu(self.L2_sub_1, 'R_conv2_1')

                # With Shortcut
                self.L2_sub_2 = tf.nn.conv2d(input=self.L2_sub_1_r, filter=self.W2_sub, strides=[1,1,1,1], padding='SAME')
                self.L2_sub_2 = self.BN(input=self.L2_sub_2, scale=True, training=self.training, name='Conv2_sub_BN_2')
                self.L2_sub_2_r = self.parametric_relu(self.L2_sub_2, 'R_conv2_2') + self.L2_sub

                # 2-2
                self.L2_sub_3 = tf.nn.conv2d(input=self.L2_sub_2_r, filter=self.W2_sub, strides=[1,1,1,1], padding='SAME')
                self.L2_sub_3 = self.BN(input=self.L2_sub_3, scale=True, training=self.training, name='Conv2_sub_BN_3')
                self.L2_sub_3_r = self.parametric_relu(self.L2_sub_3, 'R_conv2_3')

                # With Shortcut
                self.L2_sub_4 = tf.nn.conv2d(input=self.L2_sub_3_r, filter=self.W2_sub, strides=[1,1,1,1], padding='SAME')
                self.L2_sub_4 = self.BN(input=self.L2_sub_4, scale=True, training=self.training, name='Conv2_sub_BN_4')
                self.L2_sub_4_r = self.parametric_relu(self.L2_sub_4 , 'R_conv2_4') + self.L2_sub_2_r



            with tf.name_scope('conv3_x'):

                self.W3_sub = tf.get_variable(name='W3_sub', shape=[3,3,64,128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.W3_sub_1 = tf.get_variable(name='W3_sub_1', shape=[3,3,128,128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

                ####################################################################################################
                ## 2N개 Layer  : (1 Layer + 1 Shortcut Connection layer) x N개
                ## Shortcut Connection에서 차원을 늘려야 하는 경우 Projection 적용, 아닌 경우 1개 Layer를 생략하여 Shorcut Connection 구현
                ## Projection 시 Linear 하도록 활성화 함수를 쓰지 않는다.
                ## Output: 32x32
                ####################################################################################################
                # 2-1
                self.L3_sub_1 = tf.nn.conv2d(input=self.L2_sub_4_r, filter=self.W3_sub, strides=[1,2,2,1], padding='SAME')
                self.L3_sub_1 = self.BN(input=self.L3_sub_1, scale=True, training=self.training, name='Conv3_sub_BN_1')
                self.L3_sub_1_r = self.parametric_relu(self.L3_sub_1, 'R_conv3_1')

                # Projection With Shortcut
                self.L3_sub_2 = tf.nn.conv2d(input=self.L3_sub_1_r, filter=self.W3_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L3_sub_2 = self.BN(input=self.L3_sub_2, scale=True, training=self.training, name='Conv3_sub_BN_2')
                input_x = tf.layers.conv2d(inputs=self.L2_sub_4_r, kernel_size=(1,1), strides=(2,2), padding='SAME', filters=128)
                self.L3_sub_2_r = self.parametric_relu(self.L3_sub_2, 'R_conv3_2') + input_x

                # 2-2
                self.L3_sub_3 = tf.nn.conv2d(input=self.L3_sub_2_r, filter=self.W3_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L3_sub_3 = self.BN(input=self.L3_sub_3, scale=True, training=self.training, name='Conv3_sub_BN_7')
                self.L3_sub_3_r = self.parametric_relu(self.L3_sub_3, 'R_conv3_7')

                # Projection With Shortcut
                self.L3_sub_4 = tf.nn.conv2d(input=self.L3_sub_3_r, filter=self.W3_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L3_sub_4 = self.BN(input=self.L3_sub_4, scale=True, training=self.training, name='Conv3_sub_BN_8')
                self.L3_sub_4_r = self.parametric_relu(self.L3_sub_4, 'R_conv3_8') + self.L3_sub_2_r


            with tf.name_scope('conv4_x'):


                self.W4_sub = tf.get_variable(name='W4_sub', shape=[3,3,128,256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.W4_sub_1 = tf.get_variable(name='W4_sub_1', shape=[3,3,256,256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

                ####################################################################################################
                ## 2N개 Layer  : (1 Layer + 1 Shortcut Connection layer) x N개
                ## Shortcut Connection에서 차원을 늘려야 하는 경우 Projection 적용, 아닌 경우 1개 Layer를 생략하여 Shorcut Connection 구현
                ## Projection 시 Linear 하도록 활성화 함수를 쓰지 않는다.
                ## Output: 16x16
                ####################################################################################################
                # 2-1
                self.L4_sub_1 = tf.nn.conv2d(input=self.L3_sub_4_r, filter=self.W4_sub, strides=[1,2,2,1], padding='SAME')
                self.L4_sub_1 = self.BN(input=self.L4_sub_1, scale=True, training=self.training, name='Conv4_sub_BN_1')
                self.L4_sub_1_r = self.parametric_relu(self.L4_sub_1, 'R_conv4_1')

                # Projection With Shortcut
                self.L4_sub_2 = tf.nn.conv2d(input=self.L4_sub_1_r, filter=self.W4_sub_1, strides=[1,1,1,1], padding='SAME')
                self.L4_sub_2 = self.BN(input=self.L4_sub_2, scale=True, training=self.training, name='Conv4_sub_BN_2')
                input_x = tf.layers.conv2d(self.L3_sub_4_r, kernel_size=(1,1), strides=(2,2), filters=256, padding='SAME')
                self.L4_sub_2_r = self.parametric_relu(self.L4_sub_2, 'R_conv4_2') + input_x


                # 2-2
                self.L4_sub_3 = tf.nn.conv2d(input=self.L4_sub_2_r, filter=self.W4_sub_1, strides=[1,1,1,1], padding='SAME')
                self.L4_sub_3 = self.BN(input=self.L4_sub_3, scale=True, training=self.training, name='Conv4_sub_BN_11')
                self.L4_sub_3_r = self.parametric_relu(self.L4_sub_3, 'R_conv4_11')

                # With Shortcut
                self.L4_sub_4 = tf.nn.conv2d(input=self.L4_sub_3_r, filter=self.W4_sub_1, strides=[1,1,1,1], padding='SAME')
                self.L4_sub_4 = self.BN(input=self.L4_sub_4, scale=True, training=self.training, name='Conv4_sub_BN_12')
                self.L4_sub_4_r = self.parametric_relu(self.L4_sub_4, 'R_conv4_12') + self.L4_sub_2_r



            with tf.name_scope('conv5_x'):

                self.W5_sub = tf.get_variable(name='W5_sub', shape=[3,3,256,512], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.W5_sub_1 = tf.get_variable(name='W5_sub_1', shape=[3,3,512,512], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())

                ####################################################################################################
                ## 2N개 Layer  : (1 Layer + 1 Shortcut Connection layer) x N개
                ## Shortcut Connection에서 차원을 늘려야 하는 경우 Projection 적용, 아닌 경우 1개 Layer를 생략하여 Shorcut Connection 구현
                ## Projection 시 Linear 하도록 활성화 함수를 쓰지 않는다.
                ## Output: 8x8
                ####################################################################################################
                # 2-1
                self.L5_sub_1 = tf.nn.conv2d(input=self.L4_sub_4_r, filter=self.W5_sub, strides=[1,2,2,1], padding='SAME')
                self.L5_sub_1 = self.BN(input=self.L5_sub_1, scale=True, training=self.training, name='Conv5_sub_BN_1')
                self.L5_sub_1_r = self.parametric_relu(self.L5_sub_1, 'R_conv5_1')

                # Projection With Shortcut
                self.L5_sub_2 = tf.nn.conv2d(input=self.L5_sub_1_r, filter=self.W5_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L5_sub_2 = self.BN(input=self.L5_sub_2, scale=True, training=self.training, name='Conv5_sub_BN_2')
                input_x = tf.layers.conv2d(self.L4_sub_4_r, kernel_size=(1,1), strides=(2,2), filters=512, padding='SAME')
                self.L5_sub_2_r = self.parametric_relu(self.L5_sub_2, 'R_conv5_2') + input_x

                # 2-2
                self.L5_sub_3 = tf.nn.conv2d(input=self.L5_sub_2_r, filter=self.W5_sub_1, strides=[1,1,1,1], padding='SAME')
                self.L5_sub_3 = self.BN(input=self.L5_sub_3, scale=True, training=self.training, name='Conv5_sub_BN_5')
                self.L5_sub_3_r = self.parametric_relu(self.L5_sub_3, 'R_conv5_5')

                # With Shortcut
                self.L5_sub_4 = tf.nn.conv2d(input=self.L5_sub_3_r, filter=self.W5_sub_1, strides=[1, 1, 1, 1], padding='SAME')
                self.L5_sub_4 = self.BN(input=self.L5_sub_4, scale=True, training=self.training, name='Conv5_sub_BN_6')
                self.L5_sub_4_r = self.parametric_relu(self.L5_sub_4, 'R_conv5_6') + self.L5_sub_2_r


            with tf.name_scope('avg_pool'):

                ####################################################################################################
                ## Global Average Pooling
                ####################################################################################################
                self.global_avg_pool = tflearn.layers.conv.global_avg_pool(self.L5_sub_4_r, name='global_avg')

            with tf.name_scope('fc_layer1'):
                self.W_fc1 = tf.get_variable(name='W_fc1', shape=[512, 1000], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b_fc1 = tf.Variable(tf.constant(value=0.001, shape=[1000], name='b_fc1'))
                self.L6 = tf.matmul(self.global_avg_pool, self.W_fc1) + self.b_fc1
                self.L6 = self.BN(input=self.L6, scale=True, training=self.training, name='Conv6_sub_BN')
                self.L_fc1 = self.parametric_relu(self.L6, 'R_fc1')

            self.W_out = tf.get_variable(name='W_out', shape=[1000, self.class_num], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[self.class_num], name='b_out'))
            self.logits = tf.matmul(self.L_fc1, self.W_out) + self.b_out

        ################################################################################################################
        ## ▣ L2-Regularization
        ##  ⊙ λ/(2*N)*Σ(W)²-> (0.001/(2*tf.to_float(tf.shape(self.X)[0])))*tf.reduce_sum(tf.square(self.W7))
        ################################################################################################################
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + (0.01/(2*tf.to_float(tf.shape(self.Y)[0])))*tf.reduce_sum(tf.square(self.W_out))

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.005,
                                                   momentum=0.9, decay=0.99,
                                                   epsilon=0.01).minimize(self.cost)

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(self.cost)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

        # self.tensorflow_summary()

    ####################################################################################################################
    ## ▣ Tensorboard logging
    ##  ⊙ tf.summary.histogram : 여러개의 행렬 값을 logging 하는 경우
    ##  ⊙ tf.summary.scalar : 한개의 상수 값을 logging 하는 경우
    ####################################################################################################################
    def tensorflow_summary(self):
        self.W1_hist = tf.summary.histogram('W1_conv1', self.W1)
        self.b1_hist = tf.summary.histogram('b1_conv1', self.b1)
        self.L1_hist = tf.summary.histogram('L1_conv1', self.L1)

        self.W2_hist = tf.summary.histogram('W2_conv2', self.W2)
        self.b2_hist = tf.summary.histogram('b2_conv2', self.b2)
        self.L2_hist = tf.summary.histogram('L2_conv2', self.L2)

        self.W3_hist = tf.summary.histogram('W3_conv3', self.W3)
        self.b3_hist = tf.summary.histogram('b3_conv3', self.b3)
        self.L3_hist = tf.summary.histogram('L3_conv3', self.L3)

        self.W4_hist = tf.summary.histogram('W4_conv4', self.W4)
        self.b4_hist = tf.summary.histogram('b4_conv4', self.b4)
        self.L4_hist = tf.summary.histogram('L4_conv4', self.L4)

        self.W5_hist = tf.summary.histogram('W5_conv5', self.W5)
        self.b5_hist = tf.summary.histogram('b5_conv5', self.b5)
        self.L5_hist = tf.summary.histogram('L5_conv5', self.L5)

        self.W_fc1_hist = tf.summary.histogram('W6_fc1', self.W_fc1)
        self.b_fc1_hist = tf.summary.histogram('b6_fc1', self.b_fc1)
        self.L_fc1_hist = tf.summary.histogram('L6_fc1', self.L_fc1)

        self.W_fc2_hist = tf.summary.histogram('W6_fc2', self.W_fc2)
        self.b_fc2_hist = tf.summary.histogram('b6_fc2', self.b_fc2)
        self.L_fc2_hist = tf.summary.histogram('L6_fc2', self.L_fc2)

        self.cost_hist = tf.summary.scalar(self.name+'/cost_hist', self.cost)
        self.accuracy_hist = tf.summary.scalar(self.name+'/accuracy_hist', self.accuracy)

        # ※ merge_all 로 하는 경우, hist 를 모으지 않는 변수들도 대상이 되어서 에러가 발생한다.
        #    따라서 merge 로 모으고자하는 변수를 각각 지정해줘야한다.
        self.merged = tf.summary.merge([self.W1_hist, self.b1_hist, self.L1_hist,
                                        self.W2_hist, self.b2_hist, self.L2_hist,
                                        self.W3_hist, self.b3_hist, self.L3_hist,
                                        self.W4_hist, self.b4_hist, self.L4_hist,
                                        self.W5_hist, self.b5_hist, self.L5_hist,
                                        self.W_fc1_hist, self.b_fc1_hist, self.L_fc1_hist,
                                        self.W_fc2_hist, self.b_fc2_hist, self.L_fc2_hist,
                                        self.cost_hist, self.accuracy_hist])

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    ####################################################################################################################
    ## ▣ Parametric Relu or Leaky Relu
    ##  ⊙ alpha 값을 설정해 0 이하인 경우 alpha 만큼의 경사를 설정해서 0 이 아닌 값을 리턴하는 함수
    ##  ⊙ Parametric Relu : 학습을 통해 최적화된 alpha 값을 구해서 적용하는 Relu 함수
    ##  ⊙ Leaky Relu      : 0 에 근접한 작은 값을 alpha 값으로 설정하는 Relu 함수
    ####################################################################################################################
    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name,
                                 _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.01),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def BN(self, input, training, scale, name, decay=0.99):
        return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training, updates_collections=None, scope=name)