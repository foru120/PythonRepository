※ 데이터 셋
 - Tiny ImageNet
  - class: 200
  - image per class: 500 (train), 50 (test), 50 (validation)
  - size: 64 x 64

※ Tiny ImageNet 데이터 셋을 테스트 하기 위한 신경망
   (Weakly Supervised Medical Diagnosis and Localization from Multiple Resolutions)

▣ 수행 로그
 > LOG 1.
  - 축소된 mobilenet-v2 에서 bottleneck_3 의 strides 를 1로 변경하고 최종 conv2d Layer 의 출력 특징 맵 개수를 640 으로 감소
  - Train: epoch(43), acc(0.469), loss(3.136)
  - Validation: acc(0.292), loss(4.094)
  - Validation Accuracy 가 증가하지 않아 도중 훈련 종료
  - learning rate: 0.001, learning rate decay: 0.98, optimizer: SGD
  - 소스
    with tf.variable_scope(name_or_scope='input_scope'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
        self.global_step = tf.Variable(0, trainable=False)

    with tf.variable_scope(name_or_scope='body_scope'):
        layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=2, name='conv2d_0')
        layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')

        layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
        layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=4, name='bottleneck_2')
        layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=1, repeat=3, factor=4, name='bottleneck_3')
        layer = self.inverted_bottleneck(inputs=layer, filters=64, strides=2, repeat=4, factor=4, name='bottleneck_4')
        layer = self.inverted_bottleneck(inputs=layer, filters=96, strides=1, repeat=1, factor=4, name='bottleneck_5')
        layer = self.inverted_bottleneck(inputs=layer, filters=160, strides=2, repeat=3, factor=6, name='bottleneck_6')
        layer = self.inverted_bottleneck(inputs=layer, filters=320, strides=1, repeat=1, factor=6, name='bottleneck_7')

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.histogram('bottleneck_module', layer))

        layer = self.conv2d(inputs=layer, filters=640, name='conv2d_8')
        layer = self.batch_norm(inputs=layer, name='conv2d_8_batch')
        self.cam_layer = layer
        layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='conv2d_8_dropout')
        layer = tf.layers.average_pooling2d(inputs=layer, pool_size=4, strides=1, name='conv2d_8_avg_pool')
        layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='conv2d_8_output')
        self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')

    with tf.variable_scope(name_or_scope='output_scope'):
        self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

        self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
        self.loss = tf.add_n([self.loss] +
                             [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, -1), self.y), dtype=tf.float32))

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.scalar('loss', self.loss))
            self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

        self.decay_lr = tf.train.exponential_decay(self.lr, self.global_step, 1000, 0.98, staircase=True)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.decay_lr)

        update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
        with tf.control_dependencies(update_opt):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

 > LOG 2.
  - 축소된 mobilenet-v2 에서 bottleneck_3 의 strides 를 1로 변경하고 최종 conv2d Layer 의 출력 특징 맵 개수를 640 으로 감소
  - Train: epoch(56), acc(0.53), loss(2.84)
  - Validation: acc(0.309), loss(4.026)
  - Validation Accuracy 가 증가하지 않아 도중 훈련 종료
  - learning rate: 0.01, learning rate decay: 0.96, optimizer: SGD
  - 소스
    with tf.variable_scope(name_or_scope='input_scope'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
        self.global_step = tf.Variable(0, trainable=False)

    with tf.variable_scope(name_or_scope='body_scope'):
        layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=2, name='conv2d_0')
        layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')

        layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
        layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=4, name='bottleneck_2')
        layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=1, repeat=3, factor=4, name='bottleneck_3')
        layer = self.inverted_bottleneck(inputs=layer, filters=64, strides=2, repeat=4, factor=4, name='bottleneck_4')
        layer = self.inverted_bottleneck(inputs=layer, filters=96, strides=1, repeat=1, factor=4, name='bottleneck_5')
        layer = self.inverted_bottleneck(inputs=layer, filters=160, strides=2, repeat=3, factor=6, name='bottleneck_6')
        layer = self.inverted_bottleneck(inputs=layer, filters=320, strides=1, repeat=1, factor=6, name='bottleneck_7')

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.histogram('bottleneck_module', layer))

        layer = self.conv2d(inputs=layer, filters=640, name='conv2d_8')
        layer = self.batch_norm(inputs=layer, name='conv2d_8_batch')
        self.cam_layer = layer
        layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='conv2d_8_dropout')
        layer = tf.layers.average_pooling2d(inputs=layer, pool_size=4, strides=1, name='conv2d_8_avg_pool')
        layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='conv2d_8_output')
        self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')

    with tf.variable_scope(name_or_scope='output_scope'):
        self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

        self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
        self.loss = tf.add_n([self.loss] +
                             [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, -1), self.y), dtype=tf.float32))

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.scalar('loss', self.loss))
            self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

        self.decay_lr = tf.train.exponential_decay(self.lr, self.global_step, 1000, flags.FLAGS.learning_rate_decay, staircase=True)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.decay_lr)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.decay_lr)

        update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
        with tf.control_dependencies(update_opt):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

        if self.is_tb_logging:
            self.summary_merged_values = tf.summary.merge(inputs=self.summary_values)

 > LOG 3.
  - 기존 mobilenet-v2 에서 최종 conv2d Layer 의 출력 특징 맵 개수를 640 으로 감소
  - Train: epoch(40), acc(0.61), loss(2.65)
  - Validation: acc(0.300), loss(4.28)
  - Epoch 40 에서 Early Stopping 발생
  - learning rate: 0.01, learning rate decay: 0.96, optimizer: SGD
  - 소스
    with tf.variable_scope(name_or_scope='input_scope'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
        self.global_step = tf.Variable(0, trainable=False)

    with tf.variable_scope(name_or_scope='body_scope'):
        layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=1, name='conv2d_0')
        layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')

        layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
        layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
        layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=2, repeat=3, factor=6, name='bottleneck_3')
        layer = self.inverted_bottleneck(inputs=layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
        layer = self.inverted_bottleneck(inputs=layer, filters=96, strides=1, repeat=3, factor=6, name='bottleneck_5')
        layer = self.inverted_bottleneck(inputs=layer, filters=160, strides=2, repeat=3, factor=6, name='bottleneck_6')
        layer = self.inverted_bottleneck(inputs=layer, filters=320, strides=1, repeat=1, factor=6, name='bottleneck_7')

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.histogram('bottleneck_module', layer))

        layer = self.conv2d(inputs=layer, filters=640, name='conv2d_8')
        layer = self.batch_norm(inputs=layer, name='conv2d_8_batch')
        self.cam_layer = layer
        layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='conv2d_8_dropout')
        layer = tf.layers.average_pooling2d(inputs=layer, pool_size=4, strides=1, name='conv2d_8_avg_pool')
        layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='conv2d_8_output')
        self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')

    with tf.variable_scope(name_or_scope='output_scope'):
        self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

        self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
        self.loss = tf.add_n([self.loss] +
                             [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, -1), self.y), dtype=tf.float32))

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.scalar('loss', self.loss))
            self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

        self.decay_lr = tf.train.exponential_decay(self.lr, self.global_step, 1000, flags.FLAGS.learning_rate_decay, staircase=True)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.decay_lr)

        update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
        with tf.control_dependencies(update_opt):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

        if self.is_tb_logging:
            self.summary_merged_values = tf.summary.merge(inputs=self.summary_values)

 > LOG 4.
  - 기존 mobilenet-v2 에서 옵티마이저를 AdamOptimizer 로 변경
  - Train: epoch(33), acc(0.839), loss(1.222)
  - Validation: acc(0.434), loss(3.597)
  - Epoch 33 에서 Early Stopping 발생
  - learning rate: 0.0005, learning rate decay: 0.98, optimizer: Adam
  - 소스
    with tf.variable_scope(name_or_scope='input_scope'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
        self.global_step = tf.Variable(0, trainable=False)

    with tf.variable_scope(name_or_scope='body_scope'):
        layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=1, name='conv2d_0')
        layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')

        layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
        layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
        layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=2, repeat=3, factor=6, name='bottleneck_3')
        layer = self.inverted_bottleneck(inputs=layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
        layer = self.inverted_bottleneck(inputs=layer, filters=96, strides=1, repeat=3, factor=6, name='bottleneck_5')
        layer = self.inverted_bottleneck(inputs=layer, filters=160, strides=2, repeat=3, factor=6, name='bottleneck_6')
        layer = self.inverted_bottleneck(inputs=layer, filters=320, strides=1, repeat=1, factor=6, name='bottleneck_7')

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.histogram('bottleneck_module', layer))

        layer = self.conv2d(inputs=layer, filters=1280, name='conv2d_8')
        layer = self.batch_norm(inputs=layer, name='conv2d_8_batch')
        self.cam_layer = layer
        layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='conv2d_8_dropout')
        layer = tf.layers.average_pooling2d(inputs=layer, pool_size=4, strides=1, name='conv2d_8_avg_pool')
        layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='conv2d_8_output')
        self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')

    with tf.variable_scope(name_or_scope='output_scope'):
        self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

        self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
        self.loss = tf.add_n([self.loss] +
                             [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, -1), self.y), dtype=tf.float32))

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.scalar('loss', self.loss))
            self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

        self.decay_lr = tf.train.exponential_decay(self.lr, self.global_step, 1000, flags.FLAGS.learning_rate_decay, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.decay_lr)

        update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
        with tf.control_dependencies(update_opt):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

        if self.is_tb_logging:
            self.summary_merged_values = tf.summary.merge(inputs=self.summary_values)

 > LOG 5.
  - 기존 mobilenet-v2 에서 옵티마이저를 AdamOptimizer 로 변경
  - Train: epoch(79), acc(0.954), loss(0.95)
  - Validation: acc(0.309), loss(5.059)
  - Validation acc 가 높아지지 않아 학습 중단
  - early stopping patient: 100
  - Batch Normalization: zero_debias_moving_mean=True 옵션 추가
  - Epoch 33 에서 Early Stopping 발생
  - learning rate: 0.0001, learning rate decay: 0.98, optimizer: Adam
  - 소스
    with tf.variable_scope(name_or_scope='input_scope'):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='x')
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')
        self.global_step = tf.Variable(0, trainable=False)

    with tf.variable_scope(name_or_scope='body_scope'):
        layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=1, name='conv2d_0')
        layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')

        layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
        layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
        layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=2, repeat=3, factor=6, name='bottleneck_3')
        layer = self.inverted_bottleneck(inputs=layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
        layer = self.inverted_bottleneck(inputs=layer, filters=96, strides=1, repeat=3, factor=6, name='bottleneck_5')
        layer = self.inverted_bottleneck(inputs=layer, filters=160, strides=2, repeat=3, factor=6, name='bottleneck_6')
        layer = self.inverted_bottleneck(inputs=layer, filters=320, strides=1, repeat=1, factor=6, name='bottleneck_7')

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.histogram('bottleneck_module', layer))

        layer = self.conv2d(inputs=layer, filters=1280, name='conv2d_8')
        layer = self.batch_norm(inputs=layer, name='conv2d_8_batch')
        self.cam_layer = layer
        layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='conv2d_8_dropout')
        layer = tf.layers.average_pooling2d(inputs=layer, pool_size=4, strides=1, name='conv2d_8_avg_pool')
        layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='conv2d_8_output')
        self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')

    with tf.variable_scope(name_or_scope='output_scope'):
        self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

        self.prob = tf.nn.softmax(logits=self.logits, name='softmax')

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
        self.loss = tf.add_n([self.loss] +
                             [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, -1), self.y), dtype=tf.float32))

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.scalar('loss', self.loss))
            self.summary_values.append(tf.summary.scalar('accuracy', self.accuracy))

        self.decay_lr = tf.train.exponential_decay(self.lr, self.global_step, 1000, flags.FLAGS.learning_rate_decay, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.decay_lr)

        update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
        with tf.control_dependencies(update_opt):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

        if self.is_tb_logging:
            self.summary_merged_values = tf.summary.merge(inputs=self.summary_values)

 > LOG 6.
  - LOG 4에서 사용된 신경망을 그대로 사용 (단, is_training 변수를 feed_dict 형태로 변환)
  - Train: epoch(19), acc(0.634), loss(2.026)
  - Validation: acc(0.421), loss(3.243)
  - Validation acc 가 높아지지 않아 학습 중단
  - early stopping patient: 20
  - Batch Normalization: zero_debias_moving_mean=True 옵션 추가
  - learning rate: 0.0005, learning rate decay: 0.98, optimizer: Adam

 > LOG 7.
  - LOG 4에서 사용된 신경망을 그대로 사용 (Learning rate 만 변경)
  - Train: epoch(18), acc(0.622), loss(1.909)
  - Validation: acc(0.439), loss(2.942)
  - Validation acc 가 높아지지 않아 학습 중단
  - early stopping patient: 20
  - Batch Normalization: zero_debias_moving_mean=True 옵션 추가
  - learning rate: 0.001, learning rate decay: 0.98, optimizer: Adam