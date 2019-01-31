import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

from Projects.DeepLearningTechniques.PNASNet.cifar10.pnasnet_constants import *

class GradCAM:
    def __init__(self, instance, sample_size, name='grad_cam'):
        self.instance = instance
        self.sample_size = sample_size
        self.name = name
        self.image_size = (flags.FLAGS.image_width, flags.FLAGS.image_height)  # (width, height)
        self.num_classes = flags.FLAGS.image_class  # class 개수

    def normalize(self, x):
        return tf.div(x, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.sqrt(
            tf.reduce_mean(tf.square(x), axis=(1, 2, 3))) + tf.constant(1e-5), axis=-1), axis=-1), axis=-1))

    def build(self):
        '''
            self.instance.logits: 신경망에서 softmax 거치기 전 layer 출력 변수
            self.instance.prob: softmax 거쳐서 나온 확률 변수
            self.instance.cam_layer: CAM 을 통해 보고자 하는 layer
            self.instance.sess: 해당 신경망 모델에서 사용하는 Session Instance
        '''
        with tf.variable_scope(self.name):
            cam_layer = self.instance.cam_layer
            loss = tf.reduce_mean(tf.multiply(self.instance.logits, self.instance.prob), axis=-1)
            grads = tf.gradients(ys=loss, xs=cam_layer)[0]  # (B, H, W, C)
            norm_grads = self.normalize(grads)

            weights = tf.reduce_mean(input_tensor=norm_grads, axis=(1, 2))
            weights = tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1)
            height, width = cam_layer.get_shape().as_list()[1: 3]
            cam = tf.ones(shape=[self.sample_size, height, width], dtype=tf.float32)
            cam = tf.add(cam, tf.reduce_sum(input_tensor=tf.multiply(weights, cam_layer), axis=-1))
            self.cam = tf.maximum(cam, 0, name='outputs')

    def visualize(self, x, file_names):
        cam_output, cam_prob = self.instance.sess.run([self.cam, self.instance.prob],
                                                      feed_dict={self.instance.x: x})
        cam_list = []

        for idx in range(self.sample_size):
            cam_output[idx] = cam_output[idx] / np.max(cam_output[idx])
            cam_list.append(cv2.resize(cam_output[idx], self.image_size))

        outputs = []

        for idx in range(self.sample_size):
            img = Image.open(file_names[idx], mode='r').convert('RGB')
            img = cv2.resize(np.asarray(img), self.image_size, interpolation=cv2.INTER_NEAREST)
            img = img.astype(float)
            img /= 255.

            img_cam = cv2.applyColorMap(np.uint8(255 * cam_list[idx]), cv2.COLORMAP_JET)
            img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)

            '''Grad-CAM 과 원본 이미지 중첩'''
            alpha = 0.0025
            output = img + alpha * img_cam
            output /= output.max()
            outputs.append(output)

        return outputs, np.argmax(cam_prob, axis=-1)