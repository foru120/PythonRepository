■ batch_norm_v5
 - epoch: 5
    def batch_norm_wrapper(self, inputs, decay=0.9, epsilon=1e-3, name='batch_norm_wrapper'):
        with tf.variable_scope(name_or_scope=name):
            gamma = tf.Variable(tf.ones(inputs.get_shape().as_list()[-1]), name='gamma')
            beta = tf.Variable(tf.zeros(inputs.get_shape().as_list()[-1]), name='beta')
            moving_mean = tf.Variable(tf.zeros(inputs.get_shape().as_list()[-1]), trainable=False, name='moving_mean')
            moving_var = tf.Variable(tf.ones(inputs.get_shape().as_list()[-1]), trainable=False, name='moving_var')

            if self.is_training:
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
                train_mean = tf.assign(moving_mean,
                                        moving_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(moving_var,
                                       moving_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,
                                                     batch_mean, batch_var, beta, gamma, epsilon)
            else:
                return tf.nn.batch_normalization(inputs,
                                                 moving_mean, moving_var, beta, gamma, epsilon)

■ batch_norm_v6
 - epoch: 30
    def batch_norm_wrapper(self, inputs, decay=0.9, epsilon=1e-3, name='batch_norm_wrapper'):
        with tf.variable_scope(name_or_scope=name):
            gamma = tf.Variable(tf.ones(inputs.get_shape().as_list()[-1]), name='gamma')
            beta = tf.Variable(tf.zeros(inputs.get_shape().as_list()[-1]), name='beta')
            moving_mean = tf.Variable(tf.zeros(inputs.get_shape().as_list()[-1]), trainable=False, name='moving_mean')
            moving_var = tf.Variable(tf.ones(inputs.get_shape().as_list()[-1]), trainable=False, name='moving_var')

            if self.is_training:
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
                train_mean = tf.assign(moving_mean,
                                        moving_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(moving_var,
                                       moving_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,
                                                     batch_mean, batch_var, beta, gamma, epsilon)
            else:
                return tf.nn.batch_normalization(inputs,
                                                 moving_mean, moving_var, beta, gamma, epsilon)

■ batch_norm_v7
 - epoch: 10
    def batch_norm_layer(self, inputs, act=tf.nn.relu6, name='batch_norm_layer'):
        '''
            Batch Normalization
             - scale=True, scale factor(gamma) 를 사용
             - center=True, shift factor(beta) 를 사용
        '''
        with tf.variable_scope(name_or_scope=name):
            if self.is_training:
                return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, center=True, scale=True, fused=True,
                                                    updates_collections=tf.GraphKeys.UPDATE_OPS, activation_fn=act, is_training=True, scope='batch_norm')
            else:
                return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, center=True, scale=True, fused=True,
                                                    updates_collections=tf.GraphKeys.UPDATE_OPS, activation_fn=act, is_training=False, scope='batch_norm')