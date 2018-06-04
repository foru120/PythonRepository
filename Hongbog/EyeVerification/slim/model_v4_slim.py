import tensorflow as tf
import tensorflow.contrib.slim as slim

class Model:
    def __init__(self, low_res_inputs, mid_res_inputs, high_res_inputs, is_training=True, name='Model'):
        self.low_res_inputs = low_res_inputs
        self.mid_res_inputs = mid_res_inputs
        self.high_res_inputs = high_res_inputs
        self.name = name
        self.N = 12  # Dense Block 내의 Layer 개수
        self.growthRate = 10  # k
        self.compression_factor = 0.5
        self.hidden_num = 16
        self.is_training = is_training
        self.dropout_rate = 0.6
        self._build_graph()

    def _build_graph(self):
        def _batch_norm(inputs, act=tf.nn.relu6, scope='batch_norm'):
            with slim.arg_scope([slim.batch_norm], decay=0.999, epsilon=0.001, reuse=False, zero_debias_moving_mean=True, scope=scope):
                return slim.batch_norm(inputs=inputs, activation_fn=act, is_training=self.is_training)

        def _residual_block(inputs, num_outputs, kernel_size, stride, padding='SAME', scope='residual_block'):
            with tf.variable_scope(scope):
                layer = _batch_norm(inputs=inputs, scope='batch_norm_1')
                layer = slim.dropout(inputs=layer, keep_prob=self.dropout_rate, is_training=self.is_training, scope='residual_dropout_a')
                layer = slim.conv2d(inputs=layer, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding, scope='residual_conv_1')
                layer = _batch_norm(inputs=layer, scope='batch_norm_2')
                layer = slim.dropout(inputs=layer, keep_prob=self.dropout_rate, is_training=self.is_training, scope='residual_dropout_b')
                layer = slim.conv2d(inputs=layer, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding, scope='residual_conv_2')
                layer = tf.add(inputs, layer, name='residual_add')
            return layer

        def _network():
            with tf.variable_scope(self.name):
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.identity,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    '''Low Resolution Network'''
                    with tf.variable_scope('low_resolution_network'):
                        low_layer = slim.conv2d(inputs=self.low_res_inputs, num_outputs=self.hidden_num, kernel_size=5, stride=2, padding='SAME', scope='conv_1')
                        for i in range(1, 20):
                            low_layer = _residual_block(inputs=low_layer, num_outputs=self.hidden_num, kernel_size=3, stride=1, scope='residual_block_{}'.format(i))
                            if i % 4 == 0:
                                self.hidden_num *= 2
                                low_layer = slim.conv2d(inputs=low_layer, num_outputs=self.hidden_num, kernel_size=3, stride=2, padding='SAME', scope='subsampling_{}'.format(int(i/4)))
                        low_layer = tf.image.resize_nearest_neighbor(low_layer, (4, 8), name='upsampling_layer')

                    '''Middle Resolution Network'''
                    with tf.variable_scope('middle_resolution_network'):
                        self.hidden_num = 16
                        mid_layer = slim.conv2d(inputs=self.mid_res_inputs, num_outputs=self.hidden_num, kernel_size=5, stride=2, padding='SAME', scope='conv_1')
                        for i in range(1, 20):
                            mid_layer = _residual_block(inputs=mid_layer, num_outputs=self.hidden_num, kernel_size=3, stride=1, scope='residual_block_{}'.format(i))
                            if i % 4 == 0:
                                self.hidden_num *= 2
                                mid_layer = slim.conv2d(inputs=mid_layer, num_outputs=self.hidden_num, kernel_size=3, stride=2, padding='SAME', scope='subsampling_{}'.format(int(i/4)))
                        mid_layer = tf.image.resize_nearest_neighbor(mid_layer, (4, 8), name='upsampling_layer')

                    '''High Resolution Network'''
                    with tf.variable_scope('high_resolution_network'):
                        self.hidden_num = 16
                        high_layer = slim.conv2d(inputs=self.high_res_inputs, num_outputs=self.hidden_num, kernel_size=5, stride=2, padding='SAME', scope='conv_1')
                        for i in range(1, 20):
                            high_layer = _residual_block(inputs=high_layer, num_outputs=self.hidden_num, kernel_size=3, stride=1, scope='residual_block_{}'.format(i))
                            if i % 4 == 0:
                                self.hidden_num *= 2
                                high_layer = slim.conv2d(inputs=high_layer, num_outputs=self.hidden_num, kernel_size=3, stride=2, padding='SAME', scope='subsampling_{}'.format(int(i / 4)))
                        high_layer = tf.image.resize_nearest_neighbor(high_layer, (4, 8), name='upsampling_layer')

                    layer = tf.concat([high_layer, mid_layer, low_layer], axis=-1, name='concat_layer')
                    self.cam_layer = layer

                    with tf.variable_scope('output_layer'):
                        layer = slim.conv2d(inputs=layer, num_outputs=7, kernel_size=1, stride=1, padding='SAME', scope='logit')
                        layer = slim.avg_pool2d(inputs=layer, kernel_size=(4, 8), stride=1, padding='VALID', scope='global_avg_pool')
                        layer = tf.squeeze(layer, [1, 2], name=self.name+'squeeze_layer')

            self.logits = layer

        _network()