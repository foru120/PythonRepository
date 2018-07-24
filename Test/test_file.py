import tensorflow as tf
import time
from OBSKorea.model_ver4 import Model
import cv2
import os
import numpy as np
import math

#########################################################################################################################
## Data Preprocessing
#########################################################################################################################

img_dir = "F:/artery_labeled/png"
path_list = []
for (path, dirs, files) in os.walk(img_dir):
    for dir in dirs:
        if dir == 'X':
            path_list.append(path)

x_data_path = [i.replace('\\', '/')+'/X' for i in path_list]
y_data_path = [i.replace('\\', '/')+'/Y' for i in path_list]

x_data_list = []
y_data_list = []

for dir in x_data_path:
    dirc = dir.replace('\\', '/')
    X_data = os.listdir(dirc)
    x_data_list.append(X_data)

for dir in y_data_path:
    dirc = dir.replace('\\', '/')
    Y_data = os.listdir(dirc)
    y_data_list.append(Y_data)


def data_setting(dir): # grayscaling & random cropping
    patient_data = []
    for png_dir in os.listdir(dir):
        img = cv2.imread(dir+'/'+png_dir)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        width = np.random.randint(0, gray_image.shape[0] - 200)
        height = np.random.randint(0, gray_image.shape[1] - 200)
        gray_image = np.reshape(gray_image[width: width + 200, height: height+200], (1, 200, 200))
        patient_data.append(gray_image.tolist())

    return np.array(patient_data)


def dataset_setting(data_path, batch_num):
    data_list = []
    for i in range(0, int(math.ceil(len(data_path)/batch_num))):
        data_temp = []
        # print(i)
        for dir in data_path[batch_num*i : batch_num*(i+1)]:
            pre_data = data_setting(dir)
            data_temp.append(pre_data)
        data_list.append(np.array(data_temp))

    return np.array(data_list)


#########################################################################################################################
## Data Training
#########################################################################################################################

with tf.Session() as sess:
    stime = time.time()
    # Model(sess, 'Unet_est_model')
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    print('Learning Started!')

    epoch = 0
    batch_num = 5
    training_epochs = 1000

    for epochs in range(training_epochs):
        epoch_stime = time.time()
        x_dataset = dataset_setting(x_data_path, batch_num)
        y_dataset = dataset_setting(y_data_path, batch_num)

        for batch_eter in range(0, x_dataset.shape[0]):
            for mini_batch in range(0, x_dataset[0].shape[0]):
                train_x_batch = x_dataset[batch_eter][mini_batch]
                train_y_batch = y_dataset[batch_eter][mini_batch]
                # print(train_x_batch) -> (148, 1, 200, 200)
                Model.train(train_x_batch, train_y_batch)
            print(batch_eter)
        epoch_etime = time.time()
        print('Epoch : ', epochs, '|', 'Epoch_time : ', epoch_etime-epoch_stime)




import tensorflow as tf
import numpy as np
import PIL
import cv2
import os

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        # self.reg_losses = None
        self.ac_optimizer = {}
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            with tf.name_scope('input_layer') as scope:
                self.dropout_rate = tf.Variable(tf.constant(value=0.2), name='dropout_rate')
                self.training = tf.placeholder(tf.bool, name='training')
                # self.regularizer = tf.contrib.layers.l2_regularizer(0.01)

                self.X = tf.placeholder(tf.float32, [1, -1, 1, 200, 200], name='x_data')
                self.Y = tf.placeholder(tf.float32, [1, -1, 1, 200, 200], name='y_data')


                self.filter_num = 10

            """
            Convolution Layers 
            """
            with tf.name_scope('conv_layer') as scope:
                self.L1 = self.conv3d(self.X, layer_number=1, ksize=3, stride=1) # 10
                self.L2 = self.conv3d(self.L1, layer_number=2, ksize=3, stride=1) # 15
                self.L3 = self.conv3d(self.L2, layer_number=3, ksize=3, stride=1) # 30
                self.L4 = self.conv3d(self.L3, layer_number=4, ksize=3, stride=1) # 45
                self.L5 = self.conv3d(self.L4, layer_number=5, ksize=3, stride=1) # 60
                self.L6 = self.conv3d(self.L5, layer_number=6, ksize=3, stride=1) # 75
                self.L7 = self.conv3d(self.L6, layer_number=7, ksize=3, stride=1) # 90

            """
            Up-Convolution Layers
            """
            with tf.name_scope('up_conv_layer') as scope:
                self.L8 = self.upconv3d(self.L7, conv_layer_number=7, up_ksize=2, up_stride=1, c_ksize=3, c_stride=1) # 75
                self.L9 = self.upconv3d(self.L7, conv_layer_number=6, up_ksize=2, up_stride=1, c_ksize=3, c_stride=1) # 60
                self.L10 = self.upconv3d(self.L7, conv_layer_number=5, up_ksize=2, up_stride=1, c_ksize=3, c_stride=1) # 45
                self.L11 = self.upconv3d(self.L7, conv_layer_number=4, up_ksize=2, up_stride=1, c_ksize=3, c_stride=1) # 30
                self.L12 = self.upconv3d(self.L7, conv_layer_number=3, up_ksize=2, up_stride=1, c_ksize=3, c_stride=1) # 15

            """
            1x1 Convolution 
            """
            with tf.name_scope('1x1_conv_layer') as scope:
                self.L13 = self.conv3d_1x1(self.L12)

            """
            Loss Function - DICE LOSS
            """
            self.cost = self.dice_loss(self.L13, self.Y)

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.cost)

        # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    # def predict(self, x_test):
    #     return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    # def get_accuracy(self, x_test, y_test):
    #     return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    # def train(self, x_data, y_data):
    #     job = [self.cost, self.optimizer]
    #     for opt in self.ac_optimizer.values():
    #         job.append(opt)
    #     return self.sess.run(job, feed_dict={self.X: x_data, self.Y: y_data, self.training: True})


    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def BN(self, input, training):

        return tf.contrib.layers.batch_norm(input, decay=0.9, scale=True, is_training=training, updates_collections=ops.GraphKeys.UPDATE_OPS)

    def dice_loss(self, y_true, y_pred, epsilon=0.1):
        A, B, C, D, E = y_pred.get_shape()
        y_true_f = tf.reshape(y_true, shape=[A, B * C * D])
        y_pred_f = tf.reshape(y_pred, shape=[A, B * C * D])
        intersection = tf.reduce_sum(np.dot(y_true_f, y_pred_f))
        return (2. * intersection + epsilon) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)

    def conv3d(self, input, layer_number, ksize, stride):
        """
        Build 3D Convolution Layer by using tf.layers.conv3d. Using 3D Convolution, Batch_normalization, dropout, 3D maxpooling
        :param input: Input data
        :param layer_number: Integer, A number of this layer (ex. convolution layer 1 -> 1)
        :param ksize: Integer, A number of kernel_size. (or convolution filter size, ex. [3, 3, 3] filter -> 3)
        :param stride: Integer, A number of stride. (ex. [1, 1, 1] filter -> 1)
        :return: Tensor output
        """
        conv3d_l = tf.layers.conv3d(inputs=input, filters=layer_number * self.filter_num, kernel_size=[ksize, ksize, ksize], strides=[stride, stride, stride],
                                  padding='same', activation=None, use_bias=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), data_format='channels_first') # channels_first : (batch, channels, depth, height, width)
        conv3d_l = self.parametric_relu(conv3d_l, 'conv_'+str(layer_number))
        conv3d_l = self.BN(input=conv3d_l, training=self.training)
        conv3d_l = tf.layers.dropout(inputs=conv3d_l, rate=self.dropout_rate, training=self.training)
        conv3d_l = tf.nn.max_pool3d(input=conv3d_l, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        return conv3d_l

    def upconv3d(self, input, conv_layer_number, up_ksize, up_stride, c_ksize, c_stride):
        """
        Build 3D Up-Convolution(deconvolution) Layer by using tf.layers.conv3d_transpose. Using 3D Convolution Transpose, 3D Convolution, Batch_Normalization, dropout, 3D maxpooling
        :param input: Input data
        :param conv_layer_number: Integer. A number of Matching Convolution Layer for U-Net architecture.
        :param up_ksize: Intger, A number of upconvolution kernel_size. (or up convolution filter size, ex. [3, 3, 3] filter -> 3)
        :param up_stride: Integer, A number of upconvolution stride. (ex. [1, 1, 1] filter -> 1)
        :param c_ksize: Intger, A number of convolution kernel_size. (or convolution filter size, ex. [3, 3, 3] filter -> 3)
        :param c_stride: Integer, A number of convolution stride. (ex. [1, 1, 1] filter -> 1)
        :return: Tensor output
        """
        upconv3d_l = tf.layers.conv3d_transpose(inputs=input, filters=input.get_shape()[0], kernel_size=[up_ksize, up_ksize, up_ksize], strides=[up_stride, up_stride, up_stride],
                                                padding='same', activation=None, use_bias=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        upconv3d_l = tf.concat([input, upconv3d_l], axis=4)
        upconv3d_l = tf.layers.conv3d(inputs=upconv3d_l, filters=input.get_shape()[0]/(conv_layer_number-1)*(conv_layer_number-2), kernel_size=[c_ksize, c_ksize, c_ksize],
                                      strides=[c_stride, c_stride, c_stride], padding='same', activation=None, use_bias=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        upconv3d_l = self.BN(input=upconv3d_l, training=self.training)
        upconv3d_l = tf.layers.dropout(inputs=upconv3d_l, rate=self.dropout_rate, training=self.training)
        upconv3d_l = tf.nn.max_pool3d(input=upconv3d_l, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        return upconv3d_l

    def conv3d_1x1(self, input):
        """
        Build 3D Convolution Layer by using tf.layers.conv3d. Using 1x1 3D Convolution for replace FCN.
        :param input: Input data
        :return: Tensor output
        """
        conv3d_1x1_l = tf.layers.conv3d(inputs=input, filters=self.X.get_shape()[0], kernel_size=[1, 1, 1], strides=[1, 1, 1],
                                  padding='same', activation=None, use_bias=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        return conv3d_1x1_l