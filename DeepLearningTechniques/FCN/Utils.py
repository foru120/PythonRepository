import tensorflow as tf
from tensorflow.python.ops import array_ops


# Layers

def Conv2D(input, channel, kernel_size, name, padding='SAME', stride_size=1, use_bias=False):
    """
    based by tf.layers.conv2d()
    :param input: a tensor of input data
    :param channel: a number of output channel
    :param kernel_size: number or list or tuple of kernel size. if kernel_size=3, kernel will be (3, 3). if kernel_size=[1, 3], kernel will be (1, 3).
    :param stride_size: a number of stride size
    :param name: string of layer name
    :param padding: string of padding option. Same is default
    :param use_bias: a boolean of using bias option. False is default
    :param trainable: a boolean of train mode. True is default
    :return: a tensor of output
    """
    layer = tf.layers.conv2d(input, channel, kernel_size=kernel_size, strides=(stride_size, stride_size), padding=padding,
                             use_bias=use_bias, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name)
    return layer


def Deconv2D(input, channel, kernel_size, name, stride_size=2, use_bias=False):
    """
    based by tf.layers.conv2d()
    :param input: a tensor of input data
    :param channel: a number of output channel
    :param kernel_size: number or list or tuple of kernel size. if kernel_size=3, kernel will be (3, 3). if kernel_size=[1, 3], kernel will be (1, 3).
    :param stride_size: a number of stride size
    :param name: string of layer name
    :param padding: string of padding option. Same is default
    :param use_bias: a boolean of using bias option. False is default
    :param trainable: a boolean of train mode. True is default
    :return: a tensor of output
    """
    layer = tf.layers.conv2d_transpose(input, channel, kernel_size=kernel_size, strides=(stride_size, stride_size),
                                       use_bias=use_bias, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name)
    return layer


def BatchNormalization(input, training):
    """
    based by tf.layers.batch_normalization()
    :param input: a tensor of input data
    :param name: string of layer name
    :param trainable: a boolean of train mode. True is default. when test, use False.
    :return: a tensor of output
    """
    # layer = tf.layers.batch_normalization(input, name=name, trainable=trainable, training=training)
    layer = tf.contrib.layers.batch_norm(input, decay=0.9, updates_collections=tf.GraphKeys.UPDATE_OPS, scale=True, is_training=training)
    return layer


def Activation(input, type, name, drop_out_rate, training):
    """
    :param input: a tensor of input data
    :param type: string of activation function name
    :param name: string of layer name
    :param drop_out_rate: based by tf.layers.dropout, float of drop_out rate. if drop_out_rate 0.9, 90% of nodes will save.
    :param trainable: a boolean of train mode. True is default. when test, use False.
    :return: a tensor of output
    """
    if type == 'relu':
        layer = tf.nn.relu(input, name=name+'_relu')
        layer = tf.layers.dropout(layer, rate=drop_out_rate, training=training)
        return layer

    elif type == 'lrelu':
        alphas = 0.01
        pos = tf.nn.relu(input)
        neg = alphas * (input - abs(input)) * 0.5
        layer = pos + neg
        layer = tf.layers.dropout(layer, rate=drop_out_rate, training=training)
        return layer

    elif type == 'prelu':
        alphas = tf.get_variable(name, input.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(input)
        neg = alphas * (input - abs(input)) * 0.5
        layer = pos + neg
        layer = tf.layers.dropout(layer, rate=drop_out_rate, training=training)
        return layer


def Concat(down_layer, up_layer, axis, name):
    """
    :param down_layer: concatenating layer
    :param up_layer: target layer
    :param axis: concatenation target axis
    :param name: layer name
    :return: a tensor of output
    """
    layer = tf.concat([down_layer, up_layer], axis=axis, name=name)
    return layer


def SeparableConv2D(input, channel, kernel_size, name, padding='SAME', use_bias=False, stride_size=1):
    """
    based by tf.layers.separable_conv2d()
    :param input: a tensor of input data
    :param channel: a number of output channel
    :param kernel_size: a number of kernel size
    :param stride_size: a number of stride size
    :param name: string of layer name
    :param padding: string of padding option. Same is default
    :param use_bias: a boolean of using bias option. False is default
    :param trainable: a boolean of train mode. True is default
    :return: a tensor of output
    """
    layer = tf.layers.separable_conv2d(input, channel, kernel_size=(kernel_size, kernel_size), strides=(stride_size, stride_size), padding=padding, use_bias=use_bias, name=name)
    return layer


def MaxPooling2D(input, kernel_size, name, padding='SAME', stride_size=2):
    """
    based by tf.layers.max_pooling2d()
    :param input: a tensor of input data
    :param kernel_size: a number of kernel size
    :param name: string of layer name
    :param padding: string of padding option. Same is default
    :param stride_size: a number of stride size
    :return: a tensor of output
    """
    layer = tf.layers.max_pooling2d(input, pool_size=kernel_size, strides=stride_size, padding=padding, name=name+'_maxpooling')
    return layer


def GlobalAveragePooling2D(input, n_class, name):
    """
    replace Fully Connected Layer.
    https://www.facebook.com/groups/smartbean/permalink/1708560322490187/
    https://github.com/AndersonJo/global-average-pooling/blob/master/global-average-pooling.ipynb
    :param input: a tensor of input
    :param n_class: a number of classification class
    :return: class
    """
    # gap_filter = resnet.create_variable('filter', shape=(1, 1, 128, 10))
    gap_filter = tf.get_variable(name='gap_filter', shape=[1, 1, 2048, n_class], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
    layer = tf.nn.conv2d(input, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
    layer = tf.nn.avg_pool(layer, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
    layer = tf.reduce_mean(layer, axis=[1, 2])
    return layer


def Dense(input, output_channel):
    """
    fully connected layer
    :param input: a tensor of input
    :param output_channel: a number of output channel
    :param trainable: a boolean of train mode. True is default
    :return: a tensor of output
    """
    layer = tf.contrib.layers.fully_connected(inputs=input, num_outputs=output_channel)
    return layer


def iou_coe(output, target, threshold=0.5, smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, and 1 means totally match.

    Parameters
    -----------
    output : tensor
        A batch of distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.

    """
    axis = [1, 2, 3]
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou


# Loss Funtions

def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def DICE_(output, target, smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : list of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    outputs = tf.nn.softmax(network.outputs)
    dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__
    """
    axis = [1, 2, 3]
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)

    dice = (2. * inse + smooth) / (l + r + smooth)

    dice = tf.reduce_mean(dice)

    return dice


def FOCAL_(y_pred, y_true):
    scaling_factor = (1 - y_pred) ** 2
    log_pred = tf.log(y_pred)
    focal_loss = tf.reduce_sum(scaling_factor * y_true * log_pred)
    return focal_loss


def WCE_(y_pred, y_true, weight):
    wce_loss = -tf.reduce_sum(y_true[1] * tf.log(y_pred[1]) + weight * y_true[0] * tf.log(y_pred[0]))
    return wce_loss