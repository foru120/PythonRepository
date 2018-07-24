import tensorflow as tf
from Utils import *


class Nets:
    def __init__(self, batch_size, act_func, data_shape, n_class, channel=1, root_channel=6):
        # number of model root channel
        self.root_channel = root_channel

        # number of model first channel
        self.channel = channel

        # number of classification channel
        self.n_class = n_class

        # data placeholder
        self.x = tf.placeholder(tf.float32, shape=[batch_size, data_shape, data_shape, channel])
        self.y = tf.placeholder(tf.float32, shape=[batch_size, data_shape, data_shape, n_class])

        # train or test boolean
        self.training = tf.placeholder(tf.bool)

        # drop out rate
        self.keep_prob = tf.placeholder(tf.float32)

        # result
        self.logits = self.build_net()
        self.predict = tf.nn.softmax(self.logits)

        # cost
        self.cost = self._get_cost(act_func)

        # acc
        self.accuracy = iou_coe(output=self.predict, target=self.y)
        print('Model initialized')

    def _get_cost(self, act_func):
        # cross entropy loss
        if act_func == 'ce':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            return loss

        # weighted cross entropy
        if act_func == 'wce':
            loss = WCE_(y_pred=self.predict, y_true=self.y, weight=100)
            return loss

        # dice loss
        elif act_func == 'dice':
            loss = DICE_(output=self.predict, target=self.y)
            return loss

        # focal loss
        elif act_func == 'focal':
            loss = FOCAL_(y_pred=self.predict, y_true=self.y)
            return loss

        # IOU
        elif act_func == 'iou':
            loss = -IOU_(y_pred=self.logits, y_true=self.y)
            return loss

    def build_net(self):
        channel = self.root_channel  # 16
        with tf.name_scope('down_layer_1'): # 1 256 256 -> 32 256 256 -> 32 256 256 -> 32 128 128
            # channel *= 2  # 32
            x = Conv2D(self.x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='down1_conv1', padding='same')  # 256 256
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='down1_act1', training=self.training)

            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='down1_conv2', padding='same')  # 256 256
            x = BatchNormalization(x, training=self.training)
            x1 = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='down1_act2', training=self.training)
            print('x1 shape:', x1.shape)
            x = MaxPooling2D(x1, kernel_size=2, stride_size=2, padding='same', name='pool1')  # 128 128
            print('x shape:', x.shape)

        with tf.name_scope('down_layer_2'): # 32 128 128 -> 64 128 128 -> 64 128 128 -> 64 64 64
            channel *= 2  # 64
            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='down2_conv1', padding='same')  # 128 128
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='down2_act1', training=self.training)

            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='down2_conv2', padding='same')  # 128 128
            x = BatchNormalization(x, training=self.training)
            x2 = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='down2_act2', training=self.training)
            print('x2 shape:', x2.shape)
            x = MaxPooling2D(x2, kernel_size=2, stride_size=2, padding='same', name='pool2')  # 64 64
            print('x shape:', x.shape)

        with tf.name_scope('down_layer_3'): # 64 64 64 -> 128 64 64 -> 128 64 64 -> 128 32 32
            channel *= 2  # 128
            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='down3_conv1', padding='same')  # 64 64
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='down3_act1', training=self.training)

            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='down3_conv2', padding='same')  # 64 64
            x = BatchNormalization(x, training=self.training)
            x3 = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='down3_act2', training=self.training)
            print('x3 shape:', x3.shape)
            x = MaxPooling2D(x3, kernel_size=2, stride_size=2, padding='same', name='pool3')  # 32 32
            print('x shape:', x.shape)

        with tf.name_scope('up_layer_1'): # 128 32 32 -> 256 32 32 -> 256 32 32 -> 128 64 64
            channel *= 2  # 256
            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='up1_conv1', padding='same')  # 32 32
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up1_act1', training=self.training)

            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='up1_conv2', padding='same')  # 32 32
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up1_act2', training=self.training)
            print('x shape:', x.shape)
            channel //= 2  # 128
            x = Deconv2D(x, channel=channel, kernel_size=2, stride_size=2, use_bias=False, name='up1_deconv1')  # 64 64
            print('x(deconv) shape:', x.shape)
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up1_act3', training=self.training)
            x = Concat(down_layer=x, up_layer=x3, axis=3, name='up1_concat')

        with tf.name_scope('up_layer_2'): # 256 64 64 -> 128 64 64 -> 128 64 64 -> 64 128 128
            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='up2_conv1', padding='same')  # 64 64
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up2_act1', training=self.training)

            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='up2_conv2', padding='same')  # 64 64
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up2_act2', training=self.training)
            print('x shape:', x.shape)
            channel //= 2  # 64
            x = Deconv2D(x, channel=channel, kernel_size=2, stride_size=2, use_bias=False, name='up2_deconv1')  # 128 128
            print('x(deconv) shape:', x.shape)
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up2_act3', training=self.training)
            x = Concat(down_layer=x, up_layer=x2, axis=3, name='up2_concat')

        with tf.name_scope('up_layer_3'): # 128 128 128 -> 64 128 128 -> 64 128 128 -> 32 256 256
            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='up3_conv1', padding='same')  # 128 128
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up3_act1', training=self.training)

            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='up3_conv2', padding='same')  # 128 128
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up3_act2', training=self.training)
            print('x shape:', x.shape)
            channel //= 2  # 32
            x = Deconv2D(x, channel=channel, kernel_size=2, stride_size=2, use_bias=False, name='up3_deconv1')  # 256 256
            print('x(deconv) shape:', x.shape)
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='up3_act3', training=self.training)
            x = Concat(down_layer=x, up_layer=x1, axis=3, name='up3_concat')

        with tf.name_scope('out_layer'): # 64 256 256 -> 32 256 256 -> 16 256 256 -> 1 256 256
            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='out_conv1', padding='same')  # 256 256
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='out_act1', training=self.training)

            channel //= 2  # 16
            x = Conv2D(x, channel=channel, kernel_size=3, stride_size=1, use_bias=False, name='out_conv2', padding='same')  # 256 256
            x = BatchNormalization(x, training=self.training)
            x = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='out_act2', training=self.training)
            print('x shape:', x.shape)
            x = Conv2D(x, channel=2, kernel_size=1, stride_size=1, use_bias=False, name='out_1x1', padding='same')  # 256 256
            x = BatchNormalization(x, training=self.training)
            output = Activation(x, 'relu', drop_out_rate=self.keep_prob, name='output', training=self.training)
            print('output shape:', x.shape)

        return output

#a = Nets(256)


