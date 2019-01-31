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
============================================= MobileNet-v2 =============================================
 > LOG 1.
  - MobileNet v2 모델에서 conv2d_0 과 bottleneck_3 의 strides 를 1 로 설정하고 테스트 수행 (이미지 해상도가 작아서)
  - Train: epoch(60), acc(0.854978), loss(1.612652), time(7422)
  - Validation: acc(0.678800), loss(2.250133)
  - Test: acc(0.672100), loss(2.262402)
    [[701  28  58  25  15  14  18  17  77  47]
     [ 21 831   9   9   5   3  17   3  21  81]
     [ 47   4 578  71  98  49  82  33  21  17]
     [ 18  15  77 475  33 191  95  50  16  30]
     [ 29   6 101  70 522  47 104 104   7  10]
     [  7   9  67 214  33 520  54  77   7  12]
     [ 15   8  44  67  24  16 813   2   4   7]
     [ 15   2  43  45  64  59  20 721   0  31]
     [ 97  36  13  22  13   5  11   8 758  37]
     [ 33  90   7  13   2   6  16   8  23 802]]
  - 소스
    layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=1, name='conv2d_0')
    layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')

    layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
    layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
    layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=1, repeat=3, factor=6, name='bottleneck_3')
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
    layer = self.conv2d(inputs=layer, filters=10, name='conv2d_8_output')
    self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')

  > LOG 2.
   - MobileNet v2 모델에서 전반적으로 레이어 개수 및 feature map 개수를 감소시키고 테스트 수행
   - Train: epoch(143), acc(0.824756), loss(0.868663), time(13405)
   - Validation: acc(0.705400), loss(1.233003)
   - Test: acc(0.709600), loss(1.237805)
   [[777  29  71  17  14   5   8  13  36  30]
    [ 21 907   2   6   2   5  14   4   9  30]
    [ 70  12 683  58  48  46  44  29   7   3]
    [ 29   6  94 551  31 171  66  32   9  11]
    [ 36   4  97  87 553  50  94  69   5   5]
    [ 10   8  58 201  27 618  37  37   1   3]
    [ 12   9  98  56  12  24 781   2   2   4]
    [ 23   4  44  49  45  78  13 731   3  10]
    [113  37  20  18   7   4   8   3 771  19]
    [ 49 157   4  16   2  10   7  13  18 724]]
   - 소스
    layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=1, name='conv2d_0')
    layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')

    layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
    layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=4, name='bottleneck_2')
    layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=2, repeat=3, factor=4, name='bottleneck_3')
    layer = self.inverted_bottleneck(inputs=layer, filters=64, strides=2, repeat=4, factor=4, name='bottleneck_4')
    layer = self.inverted_bottleneck(inputs=layer, filters=96, strides=1, repeat=1, factor=4, name='bottleneck_5')

    if self.is_tb_logging:
        self.summary_values.append(tf.summary.histogram('bottleneck_module', layer))

    layer = self.conv2d(inputs=layer, filters=320, name='conv2d_6')
    layer = self.batch_norm(inputs=layer, name='conv2d_6_batch')
    self.cam_layer = layer
    layer = self.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, name='conv2d_6_dropout')
    layer = tf.layers.average_pooling2d(inputs=layer, pool_size=4, strides=1, name='conv2d_6_avg_pool')
    layer = self.conv2d(inputs=layer, filters=flags.FLAGS.image_class, name='conv2d_6_output')
    self.logits = tf.squeeze(input=layer, axis=[1, 2], name='logits')