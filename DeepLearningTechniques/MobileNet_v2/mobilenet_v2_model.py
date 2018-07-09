from DeepLearningTechniques.MobileNet_v2.constants import *
import tensorflow.contrib.slim as slim

class Layers:
    def __init__(self, is_training):
        self.is_training = is_training
        self.weights_init = tf.truncated_normal_initializer(stddev=0.01)
        self.regularizer = tf.contrib.layers.l2_regularizer(0.00004)
    
    def conv2d(self, inputs, filters, kernel_size=1, strides=1, padding='same', activation=tf.identity, name=None):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=padding, activation=activation, kernel_initializer=self.weights_init,
                                kernel_regularizer=self.regularizer, name=name)

    def batch_norm(self, inputs, act=tf.nn.relu6, name='batch_norm_layer'):
        '''
            Batch Normalization
             - scale=True, scale factor(gamma) 를 사용
             - center=True, shift factor(beta) 를 사용
        '''
        with tf.variable_scope(name_or_scope=name):
            return tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, center=True, scale=True, fused=True,
                                                activation_fn=act, is_training=self.is_training, scope='batch_norm')

    def group_norm(self, inputs, G=8, epsilon=1e-5, name='group_norm'):
        '''
            G=1, Layer Normalization
            G=C, Instance Normalization
        '''
        with tf.variable_scope(name_or_scope=name):
            N, H, W, C = [-1 if shape == None else shape for shape in inputs.get_shape().as_list()]
            inputs = tf.reshape(tensor=inputs, shape=[N, G, H, W, C // G])

            mean, var = tf.nn.moments(x=inputs, axes=[2, 3, 4], keep_dims=True)
            inputs = (inputs - mean) / tf.sqrt(var + epsilon)

            inputs = tf.reshape(tensor=inputs, shape=[N, H, W, C])

            gamma = tf.Variable(initial_value=1., name='gamma')
            beta = tf.Variable(initial_value=0., name='beta')

            return gamma * inputs + beta

    def depthwise_conv2d(self, inputs, kernel_size=3, strides=2, padding='SAME', depth_multiplier=1, name=None):
        # layer = tf.nn.depthwise_conv2d(input=inputs, filter=tf.shape(inputs),
        #                                strides=[1, strides, strides, 1], padding=padding, name=name)
        layer = slim.separable_conv2d(inputs=inputs, num_outputs=None, kernel_size=kernel_size, activation_fn=tf.identity,
                                      weights_initializer=self.weights_init, weights_regularizer=self.regularizer,
                                      depth_multiplier=depth_multiplier, stride=strides, padding=padding, scope=name)
        return layer

    def inverted_bottleneck(self, inputs, filters, strides, repeat, factor, name=None):
        def _mobilenet_block(inputs, input_filters, output_filters, strides, name):
            with tf.variable_scope(name_or_scope=name):
                layer = self.conv2d(inputs=inputs, filters=input_filters * factor, name='bottleneck_layer')
                layer = self.batch_norm(inputs=layer, name='bottleneck_batch')
                # layer = self.group_norm(inputs=layer, name='bottleneck_group')
                # layer = tf.nn.relu6(layer, name='bottleneck_relu')

                layer = self.depthwise_conv2d(inputs=layer, strides=strides, name='depthwise_layer')
                layer = self.batch_norm(inputs=layer, name='depthwise_batch')
                # layer = self.group_norm(inputs=layer, name='depthwise_group')
                # layer = tf.nn.relu6(layer, name='depthwise_relu')

                layer = self.conv2d(inputs=layer, filters=output_filters, activation=tf.identity, name='linear_layer')
                layer = self.batch_norm(inputs=layer, act=None, name='linear_batch')
                # layer = self.group_norm(inputs=layer, name='linear_group')
            return layer

        prev_layer = inputs
        input_filters = inputs.get_shape().as_list()[-1]

        with tf.variable_scope(name_or_scope=name):
            for idx in range(repeat):
                layer = _mobilenet_block(inputs=prev_layer, input_filters=input_filters, output_filters=filters,
                                         strides=strides, name='mobilenet_block_{}'.format(idx))

                '''inverted_bottleneck 내의 첫 번째 layer 가 strides=2 인 경우 shortcut connection 생략'''
                if idx != 0 and strides != 2:
                    if prev_layer.get_shape().as_list()[-1] != layer.get_shape().as_list()[-1]:
                        prev_layer = self.conv2d(inputs=prev_layer, filters=filters, name='residual_match_{}'.format(idx))

                    layer = tf.add(prev_layer, layer, name='residual_add_{}'.format(idx))

                '''마지막 repeat 단계는 제외'''
                if idx != repeat-1:
                    strides = 1
                    prev_layer = layer

        return layer

class Model(Layers):
    def __init__(self, sess, is_training, name):
        super(Model, self).__init__(is_training=is_training)
        self.sess = sess
        self.name = name

    def build(self):
        with tf.variable_scope(name_or_scope=self.name):
            with tf.variable_scope(name_or_scope='input_module'):
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='x')
                self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')

            with tf.variable_scope(name_or_scope='mobilenet_v2_module'):
                layer = self.conv2d(inputs=self.x, filters=32, kernel_size=3, strides=2, name='conv2d_0')
                layer = self.batch_norm(inputs=layer, name='conv2d_0_batch')
                # layer = self.group_norm(inputs=layer, name='conv2d_0_group')
                # layer = tf.nn.relu6(layer, name='conv2d_0_group')
                layer = self.inverted_bottleneck(inputs=layer, filters=16, strides=1, repeat=1, factor=1, name='bottleneck_1')
                layer = self.inverted_bottleneck(inputs=layer, filters=24, strides=2, repeat=2, factor=6, name='bottleneck_2')
                layer = self.inverted_bottleneck(inputs=layer, filters=32, strides=2, repeat=3, factor=6, name='bottleneck_3')
                layer = self.inverted_bottleneck(inputs=layer, filters=64, strides=2, repeat=4, factor=6, name='bottleneck_4')
                layer = self.inverted_bottleneck(inputs=layer, filters=96, strides=1, repeat=3, factor=6,name='bottleneck_5')
                layer = self.inverted_bottleneck(inputs=layer, filters=160, strides=2, repeat=3, factor=6, name='bottleneck_6')
                layer = self.inverted_bottleneck(inputs=layer, filters=320, strides=1, repeat=1, factor=6, name='bottleneck_7')
                layer = self.conv2d(inputs=layer, filters=1280, name='conv2d_8')
                layer = self.batch_norm(inputs=layer, name='conv2d_8_batch')
                self.cam_layer = layer
                # layer = self.group_norm(inputs=layer, name='conv2d_8_group')
                # layer = tf.nn.relu6(layer, name='conv2d_8_group')
                layer = tf.layers.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, training=False, name='conv2d_8_dropout')
                layer = tf.reduce_mean(input_tensor=layer, axis=[1, 2], keep_dims=True, name='global_pool')
                layer = tf.layers.dropout(inputs=layer, rate=flags.FLAGS.dropout_rate, training=False, name='conv2d_9_dropout')
                layer = self.conv2d(inputs=layer, filters=2, name='conv2d_9')
                self.logits = tf.squeeze(input=layer, name='squeeze')

            with tf.variable_scope('output_module'):
                self.variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if self.name in var.name]

                self.prob = tf.nn.softmax(logits=self.logits, name='softmax')
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='ce_loss'))
                self.loss = tf.add_n([self.loss] +
                                     [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.name in var.name], name='tot_loss')
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), self.y), dtype=tf.float32))

                step = tf.Variable(0, trainable=False)
                rate = tf.train.exponential_decay(learning_rate=flags.FLAGS.learning_rate, global_step=step,
                                                  decay_steps=flags.FLAGS.decay_step, decay_rate=flags.FLAGS.decay_rate)
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=rate, momentum=0.9)
                
                update_ops = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in var.name]
                with tf.control_dependencies(update_ops):
                    self.train_op = self.optimizer.minimize(self.loss, global_step=step, var_list=self.variables)

    def train(self, x, y,):
        return self.sess.run([self.accuracy, self.loss, self.train_op], feed_dict={self.x: x, self.y: y})

    def validate(self, x, y):
        return self.sess.run([self.accuracy, self.loss], feed_dict={self.x: x, self.y: y})

    def test(self, x):
        return self.sess.run(self.logit, feed_dict={self.x: x})