※ 데이터 셋
 - Tiny ImageNet
  - class: 200
  - image per class: 500 (train), 50 (test), 50 (validation)
  - size: 64 x 64

※ Tiny ImageNet 데이터 셋을 테스트 하기 위한 신경망
   (Weakly Supervised Medical Diagnosis and Localization from Multiple Resolutions)

▣ 수행 로그
 > LOG 8.
  - inception_resnet_v2 + densenet 모델 + dropout
  - Train: epoch(50), acc(0.71), loss(2.62)
  - Validation: acc(0.439), loss(4.433)
  - Epoch 50 에서 Early Stop 발생
  - early stopping patient: 20
  - learning rate: 0.001, learning rate decay: 0.98, optimizer: RMSProp, L2: 0.0005
  - 소스
   #todo Stem Layer, 28x28, 96
    with tf.variable_scope(name_or_scope='body_scope'):
        with tf.variable_scope(name_or_scope='stem_layer'):
            layer = self.conv2d_bn(inputs=self.x, filters=8, kernel_size=3, strides=1, padding='valid', act=tf.nn.elu, name='conv_01')
            layer = self.conv2d_bn(inputs=layer, filters=8, kernel_size=3, strides=1, padding='valid', act=tf.nn.elu, name='conv_02')
            layer = self.conv2d_bn(inputs=layer, filters=16, kernel_size=3, strides=1, padding='same', act=tf.nn.elu, name='conv_03')
            layer_a = self.conv2d_bn(inputs=layer, filters=32, kernel_size=1, strides=2, padding='same', act=tf.nn.elu, name='conv_04_a01')
            layer_b = self.conv2d_bn(inputs=layer, filters=32, kernel_size=3, strides=2, padding='same', act=tf.nn.elu, name='conv_04_b01')
            layer = tf.concat([layer_a, layer_b], axis=-1, name='concat_01')
            layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='dropout_01')

            layer_a = self.conv2d_bn(inputs=layer, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_05_a01')
            layer_a = self.conv2d_bn(inputs=layer_a, filters=48, kernel_size=3, strides=1, padding='valid', act=tf.nn.elu, name='conv_05_a02')
            layer_b = self.conv2d_bn(inputs=layer, filters=16, kernel_size=1, strides=1, padding='same', act=tf.nn.elu, name='conv_05_b01')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=32, kernel_size=(5, 1), strides=1, padding='same', act=tf.nn.elu, name='conv_05_b02')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=32, kernel_size=(1, 5), strides=1, padding='same', act=tf.nn.elu, name='conv_05_b03')
            layer_b = self.conv2d_bn(inputs=layer_b, filters=48, kernel_size=3, strides=1, padding='valid', act=tf.nn.elu, name='conv_05_b04')
            stem_layer = tf.concat([layer_a, layer_b], axis=-1, name='concat_02')
            stem_layer = self.dropout(inputs=stem_layer, rate=flags.FLAGS.dropout_rate, name='dropout_02')

        #todo Inception-resnet-A, 28x28, 96
        for idx in range(5):
            if idx == 0:
                layer = self.inception_resnet_A(inputs=stem_layer, name='inception_resnet_A_' + str(idx))
            else:
                layer = self.inception_resnet_A(inputs=layer, name='inception_resnet_A_' + str(idx))

        #todo Reduction-A, 14x14, 160
        red_layer_A = self.reduction_A(inputs=layer, name='reduction_A')
        red_layer_A = self.dropout(inputs=red_layer_A, rate=flags.FLAGS.dropout_rate, name='dropout_red_A')

        #todo Inception-resnet-B, 14x14, 160
        for idx in range(10):
            if idx == 0:
                layer = self.inception_resnet_B(inputs=red_layer_A, name='inception_resnet_B_' + str(idx))
            else:
                layer = self.inception_resnet_B(inputs=layer, name='inception_resnet_B_' + str(idx))

        #todo Reduction-B, 7x7, 240
        red_layer_B = self.reduction_B(inputs=layer, name='reduction_B')
        red_layer_B = self.dropout(inputs=red_layer_B, rate=flags.FLAGS.dropout_rate, name='dropout_red_B')

        #todo Inception-resnet-C, 7x7, 240
        for idx in range(5):
            if idx == 0:
                layer = self.inception_resnet_C(inputs=red_layer_B, name='inception_resnet_C_' + str(idx))
            else:
                layer = self.inception_resnet_C(inputs=layer, name='inception_resnet_C_' + str(idx))

        #todo Reduction-C, 4x4, 400
        red_layer_C = self.reduction_C(inputs=layer, name='reduction_C')
        red_layer_C = self.dropout(inputs=red_layer_C, rate=flags.FLAGS.dropout_rate, name='dropout_red_C')

        #todo Reduction-C, Upsampling
        red_layer_C = tf.image.resize_nearest_neighbor(red_layer_C, red_layer_B.get_shape()[1:3], name='red_c_resize')

        #todo Reduction-B, DenseBlock & Upsampling
        red_layer_B = self.dense_block(inputs=red_layer_B, repeat=5, name='red_b_dense_block_01')
        red_layer_B = self.dense_block(inputs=red_layer_B, repeat=5, name='red_b_dense_block_02')
        red_layer_B = tf.concat([red_layer_B, red_layer_C], axis=-1, name='red_b_concat')
        red_layer_B = self.dense_block(inputs=red_layer_B, repeat=5, name='red_b_dense_block_03')
        red_layer_B = tf.image.resize_nearest_neighbor(red_layer_B, red_layer_A.get_shape()[1:3], name='red_b_resize')

        #todo Reduction-A, DenseBlock & Upsamling
        red_layer_A = self.dense_block(inputs=red_layer_A, repeat=4, name='red_a_dense_block_01')
        red_layer_A = self.dense_block(inputs=red_layer_A, repeat=4, name='red_a_dense_block_02')
        red_layer_A = tf.concat([red_layer_A, red_layer_B], axis=-1, name='red_a_concat')
        red_layer_A = self.dense_block(inputs=red_layer_A, repeat=4, name='red_a_dense_block_03')
        red_layer_A = tf.image.resize_nearest_neighbor(red_layer_A, stem_layer.get_shape()[1:3], name='red_a_resize')

        #todo Stem, DenseBlock
        stem_layer = self.dense_block(inputs=stem_layer, repeat=3, name='stem_dense_block_01')
        stem_layer = self.dense_block(inputs=stem_layer, repeat=3, name='stem_dense_block_02')
        stem_layer = tf.concat([stem_layer, red_layer_A], axis=-1, name='stem_concat')
        stem_layer = self.dense_block(inputs=stem_layer, repeat=3, name='stem_dense_block_03')

        #todo Global Average Pooling
        self.cam_layer = stem_layer
        layer = self.dropout(inputs=stem_layer, rate=flags.FLAGS.dropout_rate, name='last_dropout')
        layer = tf.layers.average_pooling2d(inputs=layer, pool_size=28, strides=1, name='last_avg_pool')
        layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='last_conv')
        self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')

   > LOG 9.
    - inception_resnet_v2 + densenet 모델 + dropout, stride 가 2 인 conv2d 의 kernel 크기를 3x3 으로 변환
    - Train: epoch(50), acc(0.725), loss(2.449), time(57299)
    - Validation: acc(0.455), loss(4.168)
    - Epoch 53 에서 Early Stop 발생
    - early stopping patient: 20
    - learning rate: 0.001, learning rate decay: 0.98, optimizer: RMSProp, L2: 0.0005
    - 소스 (LOG 8 과) self.conv2d stride 가 2 인 layer 의 filter size 만 다름