Tensor("right/input_module/low_res_X:0", shape=(?, 60, 160, 1), dtype=float32)
Tensor("right/input_module/mid_res_X:0", shape=(?, 80, 200, 1), dtype=float32)
Tensor("right/input_module/high_res_X:0", shape=(?, 100, 240, 1), dtype=float32)
Tensor("left/input_module/low_res_X:0", shape=(?, 60, 160, 1), dtype=float32)
Tensor("left/input_module/mid_res_X:0", shape=(?, 80, 200, 1), dtype=float32)
Tensor("left/input_module/high_res_X:0", shape=(?, 100, 240, 1), dtype=float32)

Tensor("right/output_module/softmax:0", shape=(?, 7), dtype=float32)
Tensor("left/output_module/softmax:0", shape=(?, 7), dtype=float32)

▣ 001
 ■ Data Augmentation
  - Random Crop
  - Random Brightness
  - Random contrast
  - Random Hue
  - Random Saturation

 ■ Model
  - Multi-scale + MobileNet-v2 + inception-resnet-A in v2

 ■ Parameters
  - Epoch: 20
  - Dropout Rate: 0.4
  - Batch Size: 50
  - Learning Rate: 0.001
  - Regularization Scale: 0.0005

  - Batch Normalization
   - decay: 0.9
   - center=True
   - scale=True
   - fused=True
   - updates_collections=tf.GraphKeys.UPDATE_OPS

▣ 002
 ■ Data Augmentation
  - Random Crop
  - Random Brightness
  - Random contrast
  - Random Hue
  - Random Saturation

 ■ Model
  - Multi-scale + MobileNet-v2 + inception-resnet-A in v2
  - last_layer = self.conv2d(inputs=last_layer, filters=100, name='conv2d_last'), 마지막 concat 계층 뒤에 추가

 ■ Parameters
  - Epoch: 20
  - Dropout Rate: 0.4
  - Batch Size: 50
  - Learning Rate: 0.001
  - Regularization Scale: 0.0005

  - Batch Normalization
   - decay: 0.9
   - center=True
   - scale=True
   - fused=True
   - updates_collections=tf.GraphKeys.UPDATE_OPS

▣ 003
 ■ Data Augmentation
  - Random Crop
  - Random Brightness
  - Random contrast
  - Random Hue
  - Random Saturation

 ■ Model
  - Multi-scale + MobileNet-v2 + inception-resnet-A in v2

  with tf.variable_scope('multi_scale_module'):
    tot_layer = tf.concat([low_layer, mid_layer, high_layer], axis=-1, name='multiscale_concat')

    tot_layer = self.conv2d(inputs=tot_layer, filters=320, name='tot_bottleneck')
    tot_layer = self.batch_norm(inputs=tot_layer, name='tot_batch')

    a_icpt_layer = self.inception_resnet_A(inputs=tot_layer, filters=320, name='inception_module_A')
    icpt_layer_a = self.squeeze_excitation(inputs=a_icpt_layer, num_outputs=320, name='squeeze_excitation_A')
    icpt_layer_a = self.conv2d(inputs=icpt_layer_a, filters=100, name='conv2d_A')

    b_icpt_layer = self.inception_resnet_A(inputs=a_icpt_layer, filters=320, name='inception_module_B')
    icpt_layer_b = self.squeeze_excitation(inputs=b_icpt_layer, num_outputs=320, name='squeeze_excitation_B')
    icpt_layer_b = self.conv2d(inputs=icpt_layer_b, filters=100, name='conv2d_B')

    icpt_layer_c = self.inception_resnet_A(inputs=b_icpt_layer, filters=320, name='inception_module_C')
    icpt_layer_c = self.squeeze_excitation(inputs=icpt_layer_c, num_outputs=320, name='squeeze_excitation_C')
    icpt_layer_c = self.conv2d(inputs=icpt_layer_c, filters=100, name='conv2d_C')

    last_layer = tf.concat([icpt_layer_a, icpt_layer_b, icpt_layer_c], axis=-1, name='inception_concat')
    self.cam_layer = last_layer

    if self.is_logging:
        self.summary_values.append(tf.summary.histogram('output_network', tot_layer))

    last_layer = self.dropout(inputs=last_layer, rate=flags.FLAGS.dropout_rate, name='last_dropout')
    last_layer = tf.reduce_mean(last_layer, axis=[1, 2], keep_dims=True, name='reduce_mean_last')
    last_layer = self.conv2d(inputs=last_layer, filters=7, name='conv2d_output')
    self.logits = tf.squeeze(input=last_layer, axis=[1, 2], name='squeeze_output')

 ■ Parameters
  - Epoch: 20
  - Dropout Rate: 0.4
  - Batch Size: 50
  - Learning Rate: 0.001
  - Regularization Scale: 0.0005

  - Batch Normalization
   - decay: 0.9
   - center=True
   - scale=True
   - fused=True
   - updates_collections=tf.GraphKeys.UPDATE_OPS