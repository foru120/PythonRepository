import tensorflow as tf
import tensorflow.contrib.slim as slim

def local_spatial_conv2d(inputs, num_outputs, kernel_size, spatial_size,
                         downsample=False, padding='SAME', scope=None):
    ''' Local Spatial Conv2d
        :param spatial_size: It gets a list such as [H, W],
                                  or a number(int).

        -- Example
        inputs.size --> [B, H, W, C] = [B, 224, 224, C].
        spatial_size --> [2, 2] or 2.

        -- Debug
        slice -->   inputs[:, 0:112, 0:112, :]
                    inputs[:, 0:112, 112:224, :]
                    inputs[:, 112:224, 0:112, :]
                    inputs[:, 112:224, 112:224, :]
        Each slice is performed as an input to Convolution in turn.
        And Concatenate for list of those performed.

        -- Output
        net = [B, 224, 224, C]
    '''
    if len(inputs.get_shape().as_list()) != 4:
        raise ValueError('Input\'s dims is not 4.'
                         'Now dims --> %d' % len(inputs.get_shape().as_list()))
    if type(kernel_size) == int:
        kernel_size = [kernel_size, kernel_size]
    if type(spatial_size) == int:
        spatial_size = [spatial_size, spatial_size]

    stride = 2 if downsample else 1
    ib, ih, iw, ic = inputs.get_shape().as_list()
    if ih < spatial_size[0] or iw < spatial_size[1]:
        raise ValueError('Must be bigger Height/Width of inputs than spatial_size.'
                         'Now inputs --> [%d, %d]  spatial_size --> [%d, %d]'
                         % (ih, iw, spatial_size[0], spatial_size[1]))
    hs_size = ih // spatial_size[0]
    ws_size = iw // spatial_size[1]
    if ih % hs_size != 0 or iw % hs_size != 0:
        raise ValueError('Not divide Height/Width of Inputs(A) and spatial_size(B) correctly.'
                         'remainder of (A divide B) must be 0.')

    ops = []
    sub_ops = []
    idx = 0
    for h in range(spatial_size[0]):
        for w in range(spatial_size[1]):
            idx += 1
            slice = inputs[:, h * hs_size:(h + 1) * hs_size, w * ws_size:(w + 1) * ws_size, :]
            w_net = slim.conv2d(slice, num_outputs, kernel_size,
                                stride=stride, padding=padding, scope='Conv2d_3x3_%d' % idx)
            sub_ops.append(w_net)
        ops.append(tf.concat(sub_ops, axis=2))
        sub_ops.clear()
    net = tf.concat(ops, axis=1, name=scope)

    return net