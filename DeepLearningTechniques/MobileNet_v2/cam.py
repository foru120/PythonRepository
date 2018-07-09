import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.graph_editor import subgraph
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util import compat
import darkon

class GradCAM:
    def __init__(self, instance, sample_size):
        self.instance = instance
        self.sample_size = sample_size
        self.image_size = (224, 224)  # (width, height)
        self.num_classes = 2  # class 개수

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
        with tf.variable_scope('grad_cam'):
            # cam_layer = self.instance.cam_layer
            # top1 = tf.argmax(tf.reshape(self.instance.prob, [-1]))
            # loss = tf.reduce_sum(tf.multiply(self.instance.prob, tf.one_hot(top1, self.num_classes)), axis=1)  # (B, C) -> (B, )
            # grads = tf.gradients(ys=loss, xs=cam_layer)[0]  # (B, H, W, C)
            # norm_grads = self.normalize(grads)
            # # norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)  # normalize
            #
            # weights = tf.reduce_mean(input_tensor=norm_grads, axis=(1, 2))
            # weights = tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1)
            # height, width = cam_layer.get_shape().as_list()[1: 3]
            # cam = tf.ones(shape=[self.sample_size, height, width], dtype=tf.float32)
            # cam = tf.add(cam, tf.reduce_sum(input_tensor=tf.multiply(weights, cam_layer), axis=-1))
            # self.cam = tf.maximum(cam, 0, name='outputs')

            cam_layer = self.instance.cam_layer
            loss = tf.reduce_mean(tf.multiply(self.instance.logits, self.instance.prob), axis=1)
            grads = tf.gradients(ys=loss, xs=cam_layer)[0]  # (B, H, W, C)
            norm_grads = self.normalize(grads)

            weights = tf.reduce_mean(input_tensor=norm_grads, axis=(1, 2))
            weights = tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1)
            height, width = cam_layer.get_shape().as_list()[1: 3]
            cam = tf.ones(shape=[self.sample_size, height, width], dtype=tf.float32)
            cam = tf.add(cam, tf.reduce_sum(input_tensor=tf.multiply(weights, cam_layer), axis=-1))
            self.cam = tf.maximum(cam, 0, name='outputs')

    def visualize(self, x, file_names):
        cam_output = self.instance.sess.run(self.cam,
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

        return outputs

# def guided_grad(grad):
#     return tf.where(0. < grad, grad, tf.zeros_like(grad))
#
# @ops.RegisterGradient("GuidedRelu6")
# def _guided_grad_relu6(op, grad):
#     return guided_grad(gen_nn_ops._relu6_grad(grad, op.outputs[0]))


# @ops.RegisterGradient("GuideRelu")
# def _GuidedReluGrad(op, grad):
#     return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

class GuidedGradCAM:
    def __init__(self, instance, sample_size):
        self.instance = instance
        self.sample_size = sample_size
        self.image_size = (224, 224)  # (width, height)

    def replace_grad_to_guided_grad(self, g):
        sgv = subgraph.make_view(g)
        with g.gradient_override_map({'Relu6': 'GuideRelu6'}):
            for op in sgv.ops:
                self._replace_grad(g, op)

    def _replace_grad(self, g, op):
        # ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py
        # tf.Graph._gradient_override_map
        try:
            op_def = op._op_def
            node_def = op._node_def

            if op_def is not None:
                mapped_op_type = g._gradient_override_map[op_def.name]
                node_def.attr["_gradient_op_type"].CopyFrom(
                    attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
        except KeyError:
            pass

    def normalize(self, x):
        return tf.div(x, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.sqrt(
            tf.reduce_mean(tf.square(x), axis=(1, 2, 3))) + tf.constant(1e-5), axis=-1), axis=-1), axis=-1))

    def build(self):
        '''
            self.instance.y: 신경망에서 사용하는 정답 라벨 변수 (onehot-encoding 으로 변환해야함)
            self.instance.prob: softmax 거쳐서 나온 확률 변수
        '''
        # with tf.get_default_graph().gradient_override_map({'Relu6': 'GuideRelu'}):
        #     self.instance.build_graph()
        cam_layer = self.instance.cam_layer
        loss = tf.reduce_mean(tf.multiply(self.instance.logits, self.instance.prob), axis=1)
        grads = tf.gradients(ys=loss, xs=cam_layer)[0]  # (B, H, W, C)
        norm_grads = self.normalize(grads)

        max_output = tf.reduce_max(cam_layer, axis=2)
        self.saliency_map = tf.gradients(tf.reduce_sum(max_output), self.instance.x)[0]

        weights = tf.reduce_mean(input_tensor=norm_grads, axis=(1, 2))
        weights = tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1)
        height, width = cam_layer.get_shape().as_list()[1: 3]
        cam = tf.ones(shape=[self.sample_size, height, width], dtype=tf.float32)
        cam = tf.add(cam, tf.reduce_sum(input_tensor=tf.multiply(weights, cam_layer), axis=-1))
        cam = tf.maximum(cam, 0, name='outputs')
        self.cam = tf.div(cam, tf.reduce_max(cam))

    def backpropagation(self, x):
        saliency_val = self.instance.sess.run(self.saliency_map, feed_dict={self.instance.x: x,
                                                                            self.instance.is_training: False})
        return saliency_val

    def visualize(self, x, file_names):
        '''
            self.instance.logits: 신경망에서 softmax 거치기 전 layer 출력 변수
            self.instance.prob: softmax 거쳐서 나온 확률 변수
            self.instance.cam_layer: CAM 을 통해 보고자 하는 layer
            self.instance.sess: 해당 신경망 모델에서 사용하는 Session Instance
        '''
        cam_output, saliency_val = self.instance.sess.run([self.cam, self.saliency_map],
                                                          feed_dict={self.instance.x: x,
                                                                     self.instance.training: False})
        cam = np.maximum(cam_output, 0)
        cam_list = []

        for idx in range(self.sample_size):
            cam[idx] = cam[idx] / np.max(cam[idx])
            cam_list.append(cv2.resize(cam[idx], self.image_size))

        cam_list = np.asarray(cam_list)[..., None] * saliency_val

        for idx in range(self.sample_size):
            cam_list[idx] -= np.mean(cam_list[idx])
            cam_list[idx] /= (np.std(cam_list[idx]) + 1e-5)
            cam_list[idx] *= 0.1

            cam_list[idx] += 0.5
            cam_list[idx] = np.clip(cam_list[idx], 0, 1)

            cam_list[idx] *= 255
            cam_list[idx] = np.clip(cam_list[idx], 0, 255).astype('uint8')
            cam_list[idx] = cv2.cvtColor(cam_list[idx], cv2.COLOR_BGR2RGB)

        return cam_list