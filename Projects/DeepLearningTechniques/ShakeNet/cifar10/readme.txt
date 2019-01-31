※ 데이터 셋
 - CIFAR-10
  - class: 10
  - image per class: 50,000 (train), 10,000 (test)
  - size: 32 x 32
  - label: 0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse, 8-ship, 9-truck

※ CIFAR-10 데이터 셋을 테스트 하기 위한 신경망
   (Weakly Supervised Medical Diagnosis and Localization from Multiple Resolutions)

▣ 특이사항
2018-10-29
 - 매 epoch 마다 같은 이미지가 반복되는지 확인 (Data Loader 부분) # 확인 완료

2018-10-30
 - validation 수행 시 클래스 별 이미지 개수가 맞지 않는 현상 # 확인 완료

2018-11-01
 - CIFAR-10 Dataset 비율 재 설정 (4500/1000/500)

▣ 수행 로그
============================================= Shakenet =============================================
  > LOG 1.
   - shakenet 모델 테스트 수행 [Train(4500)/Validation(500)/Test(1000)]
   - Train: epoch(48), acc(0.995), loss(0.203), time()
   - Validation: acc(0.915), loss(0.499)
   - Test: acc(0.904), loss(0.556)
   - 소스
    with tf.variable_scope(name_or_scope='body_scope'):
        x = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=1, name='conv2d_0')
        x = self.batch_norm(inputs=x, name='conv2d_0_batch')

        #todo shake stage-1
        x = self.shake_stage(x=x, output_filters=32 * self.k, num_blocks=self.num_blocks, strides=1, name='shake_stage_1')

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.histogram('shake_stage-1', x))

        #todo shake stage-2
        x = self.shake_stage(x=x, output_filters=32 * 2 * self.k, num_blocks=self.num_blocks, strides=2, name='shake_stage_2')

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.histogram('shake_stage-2', x))

        # todo shake stage-3
        x = self.shake_stage(x=x, output_filters=32 * 4 * self.k, num_blocks=self.num_blocks, strides=2, name='shake_stage_3')

        if self.is_tb_logging:
            self.summary_values.append(tf.summary.histogram('shake_stage-3', x))

        x = tf.nn.relu(x, name='relu_5')
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False, name='global_avg_pool_5')
        self.logits = self.dense(inputs=x, units=flags.FLAGS.image_class, name='dense_5')

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

        self.decay_lr = tf.train.exponential_decay(self.lr, self.global_step, 450, flags.FLAGS.learning_rate_decay, staircase=True)

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.decay_lr, momentum=0.9, use_nesterov=True)

        update_opt = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
        with tf.control_dependencies(update_opt):
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.variables, global_step=self.global_step)

   > LOG 2.
    - 소스는 LOG 3 과 동일 [Train(5000)/Test(1000)]
    - Train: epoch(48), acc(0.996), loss(0.197), time(12296)
    - Validation: acc(0.921), loss(0.480)
    - Test: acc(0.9097), loss(0.522)
    - Epoch 48 에서 Early Stopping

   > LOG 3.
    - 기존 LOG 3 소스에 cosine_decay_restarts 로 learning rate decay 부분만 변환
    - Train: total epoch(261), max => epoch(153), acc(0.999), loss(0.168), time(244.749)
    - Validation: max => epoch(155), acc(0.939), loss(0.428)
    - Test: acc(0.902), loss(0.688)
    - Epoch 261 에서 Early Stopping

   > LOG 4.
    - 기존 LOG 5 에서 Early Stopping 을 제거
    - optimizer learning rate decay: tf.train.cosine_decay_restarts
    - batch size: 128
    - Train: total epoch(390), max => epoch(314), acc(1), loss(0.114), time(114884)
    - Validation: max => epoch(307), acc(0.942), loss(0.398)
    - Test: epoch(324), acc(0.933), loss(0.437)

   > LOG 5.
    - 기존 LOG 5 에서 Early Stopping 을 제거
    - optimizer learning rate decay: tf.train.cosine_decay
    - batch size: 128
    - Train: total epoch(1800), max => epoch(813), acc(1), loss(0.058), time()
    - Validation: max => epoch(877), acc(0.948), loss(0.332)
    - Test: epoch(877), acc(0.935), loss(0.405)

   > LOG 6.
    - 기존 LOG 5 에서 random crop augmentation padding 으로 변환
    - 기존 LOG 5 에서 image whitening 추가
    - optimizer learning rate decay: tf.train.cosine_decay
    - batch size: 128
    - Train: total epoch(1033), max => epoch(809), acc(1), loss(0.058), time()
    - Validation: max => epoch(941), acc(0.951), loss(0.302)

   > LOG 7.
    - 기존 LOG 6 과 모든 설정은 동일하되 original cifar10 사용
    - optimizer learning rate decay: tf.train.cosine_decay
    - batch size: 128
    - Train: total epoch(782), max => epoch(782), acc(0.999), loss(0.075), time()
    - Validation: max => epoch(791), acc(0.938), loss(0.393)

   > LOG 8.
    - 기존 LOG 8 에서 마지막에 dropout 추가
    - dataset: original cifar10 사용
    - optimizer learning rate decay: tf.train.cosine_decay
    - batch size: 128
    - Train: total epoch(1313), max => epoch(823), acc(1), loss(0.064)
    - Validation: max => epoch(848), acc(0.945), loss(0.374)

   > LOG 9.
    - 기존 LOG 8 에서 filter_num=24 로 설정
    - dataset: original cifar10 사용
    - optimizer learning rate decay: tf.train.cosine_decay
    - batch size: 128
    - Train: total epoch(335), max => epoch(327), acc(0.921), loss(0.711)
    - Validation: max => epoch(321), acc(0.898), loss(0.831)

   > LOG 10.
    - 기존 LOG 8 에서 filter_num=30 로 설정
    - dataset: original cifar10 사용
    - optimizer learning rate decay: tf.train.cosine_decay
    - batch size: 128
    - Train: total epoch(), max => epoch(), acc(), loss()
    - Validation: max => epoch(), acc(), loss()