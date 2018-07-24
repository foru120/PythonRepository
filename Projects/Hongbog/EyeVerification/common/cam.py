import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

class CAM:
    def __init__(self, instance, sample_size):
        self.instance = instance
        self.sample_size = sample_size
        self.image_size = (80, 200)  # (height, width)

    def visualize(self, low_res_X, mid_res_X, high_res_X, labels):
        '''
             self.instance.cam_layer: GAP(Global Average Pooling) 전에 마지막 conv layer
             cam_fc_value: GAP 후에 최종 logit 을 출력하는 FC 계층의 가중치 값
             labels: 신경망 입력에 대한 label 값들
        '''
        channel = self.instance.cam_layer.get_shape().as_list()[-1]
        cam_conv_resize = tf.image.resize_images(self.instance.cam_layer, self.image_size)
        with tf.variable_scope(name_or_scope='output_part', reuse=True):
            cam_fc_value = tf.nn.bias_add(tf.get_variable('fully_connected/weights'), tf.get_variable('fully_connected/biases'))

        cam_heatmap = tf.matmul(tf.reshape(cam_conv_resize, (-1, self.image_size[0] * self.sample_size[1], channel)),
                                cam_fc_value)
        cam_heatmap = tf.reshape(cam_heatmap, shape=(-1, self.image_size[0], self.image_size[1], cam_fc_value.get_shape().as_list()[-1]))

        outputs = self.instance.sess(cam_heatmap, feec_dict={self.instance.low_res_X: low_res_X,
                                                             self.instance.mid_res_X: mid_res_X,
                                                             self.instance.high_res_X: high_res_X,
                                                             self.instance.is_training: False})

        return np.transpose(np.squeeze(outputs[:self.sample_size, :, :, labels]), axes=(0, 2, 1))  # (batch, width, height)

class GradCAM:
    def __init__(self, instance, sample_size):
        self.instance = instance
        self.sample_size = sample_size
        self.image_size = (200, 80)  # (width, height)

    def visualize(self, low_res_X, mid_res_X, high_res_X, file_names):
        '''
            self.instance.logits: 신경망에서 softmax 거치기 전 layer 출력 변수
            self.instance.prob: softmax 거쳐서 나온 확률 변수
            self.instance.cam_layer: CAM 을 통해 보고자 하는 layer
            self.instance.sess: 해당 신경망 모델에서 사용하는 Session Instance
        '''
        cam_layer = self.instance.cam_layer
        signal = tf.multiply(self.instance.logits, self.instance.prob)
        loss = tf.reduce_mean(signal)
        grads = tf.gradients(ys=loss, xs=cam_layer)[0]  # (N, H, W, C)

        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
        output, grads_val = self.instance.sess.run([cam_layer, norm_grads],
                                                   feed_dict={self.instance.low_res_X: low_res_X,
                                                              self.instance.mid_res_X: mid_res_X,
                                                              self.instance.high_res_X: high_res_X,
                                                              self.instance.is_training: False})
        weights = np.mean(grads_val, axis=(1, 2))  # (N, C)
        weights = weights[:, np.newaxis, np.newaxis, :]
        cam = np.ones(output.shape[0: 3], dtype=np.float32)  # (N, H, W)
        cam += np.sum(weights * output, axis=-1)

        cam = np.maximum(cam, 0)
        cam_list = []

        for idx in range(self.sample_size):
            cam[idx] = cam[idx] / np.max(cam[idx])
            cam_list.append(cv2.resize(cam[idx], self.image_size))

        outputs = []

        for idx in range(self.sample_size):
            img = Image.open(file_names[idx], mode='r').convert('RGB')
            img = np.asarray(img.resize(self.image_size), dtype=np.float32)
            img /= 255.

            img_cam = cv2.applyColorMap(np.uint8(255 * cam_list[idx]), cv2.COLORMAP_JET)
            img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)

            '''Grad-CAM 과 원본 이미지 중첩'''
            alpha = 0.0025
            output = img + alpha * img_cam
            output /= output.max()
            outputs.append(output)

        return outputs

class GuidedGradCAM:
    def __init__(self, instance, sample_size):
        self.instance = instance
        self.sample_size = sample_size
        self.image_size = (200, 80)  # (width, height)
        self._build_backpropagation()

    @ops.RegisterGradient("GuideRelu")
    def _GuidedReluGrad(op, grad):
        return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

    def _build_backpropagation(self):
        '''
            self.instance.y: 신경망에서 사용하는 정답 라벨 변수 (onehot-encoding 으로 변환해야함)
            self.instance.prob: softmax 거쳐서 나온 확률 변수
        '''
        with tf.get_default_graph().gradient_override_map({'Relu6': 'GuideRelu'}):
            self.instance.build_graph()

            cost = (-1) * tf.reduce_sum(tf.multiply(tf.one_hot(self.instance.y, 7), tf.log(self.instance.prob)), axis=1)
            self.grads = tf.gradients(cost, self.instance.mid_res_X)[0]

    def backpropagation(self, low_res_X, mid_res_X, high_res_X, labels):
        def _guided_backprop(grads):
            grads = np.squeeze(grads)

            # normalize
            grads -= grads.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
            grads /= (grads.std(axis=(1, 2)) + 1e-5)[:, np.newaxis, np.newaxis]
            grads *= 0.1

            # clip ~ [0, 1]
            grads = np.clip(grads + 0.5, 0, 1)

            # convert to RGB array
            grads = np.clip(grads * 255, 0, 255).astype('uint8')
            return grads

        grads = self.instance.sess.run(self.grads, feed_dict={self.instance.low_res_X: low_res_X,
                                                              self.instance.mid_res_X: mid_res_X,
                                                              self.instance.high_res_X: high_res_X,
                                                              self.instance.y: labels,
                                                              self.instance.is_training: False})
        grads = _guided_backprop(grads)

        return grads

    def visualize(self, low_res_X, mid_res_X, high_res_X, labels, grad_outputs):
        '''
            self.instance.logits: 신경망에서 softmax 거치기 전 layer 출력 변수
            self.instance.prob: softmax 거쳐서 나온 확률 변수
            self.instance.cam_layer: CAM 을 통해 보고자 하는 layer
            self.instance.sess: 해당 신경망 모델에서 사용하는 Session Instance
        '''
        cam_layer = self.instance.cam_layer
        signal = tf.reduce_sum(tf.multiply(self.instance.logits, tf.one_hot(self.instance.y, 7)), axis=1)
        grads = tf.gradients(ys=signal, xs=cam_layer)[0]  # (N, H, W, C)

        output, grads_val = self.instance.sess.run([cam_layer, grads],
                                                   feed_dict={self.instance.low_res_X: low_res_X,
                                                              self.instance.mid_res_X: mid_res_X,
                                                              self.instance.high_res_X: high_res_X,
                                                              self.instance.y: labels,
                                                              self.instance.is_training: False})
        weights = np.mean(grads_val, axis=(1, 2))  # (N, C)
        weights = weights[:, np.newaxis, np.newaxis, :]
        cam = np.ones(output.shape[0: 3], dtype=np.float32)  # (N, H, W)
        cam += np.sum(weights * output, axis=-1)

        cam = np.maximum(cam, 0)
        cam_list = []

        for idx in range(self.sample_size):
            cam[idx] = cam[idx] / np.max(cam[idx])
            cam_list.append(cv2.resize(cam[idx], self.image_size))

        cam_list = np.transpose(np.asarray(cam_list), axes=[0, 2, 1])

        return cam_list * grad_outputs