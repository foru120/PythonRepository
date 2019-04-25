import re

from Projects.Hongbog.MultiGPU.NCCL.model import Model
from Projects.Hongbog.MultiGPU.NCCL.constants import *
from tensorflow.contrib import nccl

class MultiGPU:
    def __init__(self):
        self.model = Model()

    def init_tower(self, num_gpus, batch_queue, opt):
        grad_list = []

        for gpu_idx in range(num_gpus):
            with tf.device(tf.DeviceSpec(device_type='GPU', device_index=gpu_idx)), tf.variable_scope('tower%d' % gpu_idx):
                with tf.name_scope('%s_%d' % (flags.FLAGS.tower_name, gpu_idx)) as scope:
                    train_x, train_y = batch_queue.dequeue()

                    logits = self.model.build_graph(x_batch=train_x)

                    acc, loss = self._tower_metrics(x_batch=logits, y_batch=train_y, scope=scope)

                    grad_list.append([x for x in opt[gpu_idx].compute_gradients(loss, colocate_gradients_with_ops=True) if x[0] is not None])

        grads, all_vars = self._split_grad_list(grad_list)
        reduced_grad = self._allreduce_grads(grads, average=True)
        self.grads = self._merge_grad_list(reduced_grad, all_vars)

        return acc, loss

    def _tower_metrics(self, x_batch, y_batch, scope):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_batch, logits=x_batch, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        ce_loss = tf.get_collection('losses', scope)
        l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
        tot_loss = tf.add_n(ce_loss + l2_loss, name='tot_loss')

        for l in ce_loss + [tot_loss]:
            loss_name = re.sub('%s_[0-9]*/' % flags.FLAGS.tower_name, '', l.op.name)
            tf.summary.scalar(loss_name, l)

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x_batch, -1), y_batch), dtype=tf.float32))

        return acc, tot_loss

    def _allreduce_grads(self, all_grads, average=True):
        nr_tower = len(all_grads)

        if nr_tower == 1:  # GPU 를 한개만 사용하는 경우
            return all_grads

        new_all_grads = []
        for grads in zip(*all_grads):
            summed = nccl.all_sum(grads)

            grads_for_devices = []
            for g in summed:
                with tf.device(g.device):
                    if average:
                        g = tf.multiply(g, 1.0 / nr_tower, name='allreduce_avg')
                grads_for_devices.append(g)
            new_all_grads.append(grads_for_devices)

        ret = list(zip(*new_all_grads))
        return ret

    def _split_grad_list(self, grad_list):
        """
        Args:
            grad_list: K x N x 2
        Returns:
            K x N: gradientsExiting with failure status due to previous errors
            K x N: variables
        """
        g = []
        v = []
        for tower in grad_list:
            g.append([x[0] for x in tower])
            v.append([x[1] for x in tower])

        return g, v

    def _merge_grad_list(self, all_grads, all_vars):
        """
        Args:
            all_grads (K x N): gradients
            all_vars(K x N): variables
        Return:
            K x N x 2: list of list of (grad, var) pairs
        """

        return [list(zip(gs, vs)) for gs, vs in zip(all_grads, all_vars)]