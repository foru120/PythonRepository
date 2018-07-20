import tensorflow as tf
import tensorflow.contrib.slim as slim

class NoOpScope(object):
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def safe_arg_scope(funcs, **kwargs):
    """ Returns 'slim.arg_scope' with all None arguments removed. """
    filtered_arg = {name: value for name, value in kwargs.items()
                    if value is not None}
    if filtered_arg:
        return slim.arg_scope(funcs, **filtered_arg)
    else:
        NoOpScope()

def depthwise_separable_conv(inputs, num_outputs, kernel_size, stride=1,
                             depth_multiplier=1.0, scope=None):
    with tf.variable_scope(scope, 'Separable_conv', [inputs]):
        num_outputs = round(depth_multiplier * num_outputs)
        net = slim.separable_conv2d(inputs, None, kernel_size, depth_multiplier,
                                    stride, scope='dw')
        net = slim.conv2d(net, num_outputs, [1, 1], scope='pw')

        return net

def training_scope(is_training=True, weight_decay=0.00004,
                   stddev=0.09, dropout_keep_prob=0.8, bn_decay=0.001):

    batch_norm_params = {'decay': bn_decay, 'is_training': is_training,
                         'scale': True, 'center': True}

    if stddev < 0:
        weight_initializer = slim.xavier_initializer()
    else:
        weight_initializer = tf.truncated_normal_initializer(stddev=stddev)

    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d],
                        weights_initializer=weight_initializer,
                        normalizer_fn=slim.batch_norm,
                        biases_initializer=None), \
        slim.arg_scope([mobilenet_v1], is_training=is_training), \
        safe_arg_scope([slim.batch_norm], **batch_norm_params), \
        safe_arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob), \
        slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)), \
        slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as sc:

        return sc

@slim.add_arg_scope
def mobilenet_v1(inputs, prediction_fn=slim.softmax, reuse=None,
                 scope='MobileNet', is_training=False):

    name = 'input'
    net = tf.identity(inputs, name=name)

    with tf.variable_scope(scope, 'MobileNet', [inputs], reuse=reuse):
        end_points = {}

        name = 'conv1'
        net = slim.conv2d(net, 32, [3, 3], stride=2, scope=name)
        end_points[name] = net

        name = 'separable_conv2'
        net = depthwise_separable_conv(net, 64, [3, 3], scope=name)
        end_points[name] = net

        name = 'separable_conv3'
        net = depthwise_separable_conv(net, 128, [3, 3], stride=2, scope=name)
        end_points[name] = net

        name = 'separable_conv4'
        net = depthwise_separable_conv(net, 128, [3, 3], scope=name)
        end_points[name] = net

        name = 'separable_conv5'
        net = depthwise_separable_conv(net, 256, [3, 3], stride=2, scope=name)
        end_points[name] = net

        name = 'separable_conv6'
        net = depthwise_separable_conv(net, 256, [3, 3], scope=name)
        end_points[name] = net

        name = 'separable_conv7'
        net = depthwise_separable_conv(net, 512, [3, 3], stride=2, scope=name)
        end_points[name] = net

        name = 'separable_conv8x5'
        net = slim.repeat(net, 5, depthwise_separable_conv, 512, [3, 3], scope=name)
        end_points[name] = net

        name = 'separable_conv9'
        net = depthwise_separable_conv(net, 1024, [3, 3], stride=2, scope=name)
        end_points[name] = net

        name = 'separable_conv10'
        net = depthwise_separable_conv(net, 1024, [3, 3], scope=name)
        end_points[name] = net

        with tf.variable_scope('Logits'):
            name = 'global_pool'
            net = slim.avg_pool2d(net, [6, 6], stride=1, scope=name)
            end_points[name] = net

            net = slim.dropout(net, is_training=is_training, scope='dropout')

            name = 'logits'
            logits = slim.conv2d(net, 2, [1, 1],
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 biases_initializer=tf.zeros_initializer(),
                                 scope=name)

            logits = tf.squeeze(logits, [1, 2])
            logits = tf.identity(logits, name='output')

        end_points[name] = logits
        if prediction_fn:
            name = 'predictions'
            end_points[name] = prediction_fn(logits, 'Predictions')

    return logits, end_points

def mobilenet_losses(logits, labels, scope=None):
    with tf.variable_scope(scope, 'Losses', [logits, labels]):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=labels,
                                                                      name='losses'))
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()
        return total_loss

def mobilenet_optimizer(losses, learning_rate=0.001, step=100,
                        momentum=0.9, decay_rate=0.96, scope=None):
    with tf.variable_scope(scope, 'Optimizer', [losses]):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(learning_rate, global_step,
                                        step, decay_rate, staircase=True)
        optimizer = tf.train.RMSPropOptimizer(lr, momentum=momentum)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(losses, global_step)
        # train_op = tf.train.RMSPropOptimizer(lr, momentum=momentum).minimize(losses, global_step)
        return train_op