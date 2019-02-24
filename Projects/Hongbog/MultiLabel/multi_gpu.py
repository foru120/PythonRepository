import re

from Projects.Hongbog.MultiLabel.model import Model
from Projects.Hongbog.MultiLabel.constants import *

class MultiGPU:
    def __init__(self):
        self.model = Model()

    def init_tower(self, num_gpus, batch_queue, opt):
        tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (flags.FLAGS.tower_name, i)) as scope:
                        train_x, train_y = batch_queue.dequeue()

                        logits = self.model.build_graph(x_batch=train_x)

                        acc, loss = self._tower_metrics(x_batch=logits, y_batch=train_y, scope=scope)

                        tf.get_variable_scope().reuse_variables()

                        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)

                        tower_grads.append(grads)

        self.grads = self._avg_grads(tower_grads=tower_grads)

        for grad, var in self.grads:
            if grad is not None:
                self.summaries.append(tf.summary.histogram(var.op.name + '/grad', grad))

        for var in tf.trainable_variables():
            self.summaries.append(tf.summary.histogram(var.op.name, var))

        return acc, loss

    def _tower_metrics(self, x_batch, y_batch, scope):
        binary_crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_batch, logits=x_batch, name='sigmoid_binary_cross_entropy'
        )
        binary_crossentropy_mean = tf.reduce_mean(binary_crossentropy, name='binary_cross_entropy')
        tf.add_to_collection('losses', binary_crossentropy_mean)

        ce_loss = tf.get_collection('losses', scope)
        l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
        tot_loss = tf.add_n(ce_loss + l2_loss, name='tot_loss')

        for l in ce_loss + [tot_loss]:
            loss_name = re.sub('%s_[0-9]*/' % flags.FLAGS.tower_name, '', l.op.name)
            tf.summary.scalar(loss_name, l)

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(x_batch)), y_batch), dtype=tf.float32))

        # acc = tf.reduce_mean(tf.reduce_min(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(x_batch)), y_batch), dtype=tf.float32), 1))

        return acc, tot_loss

    def _avg_grads(self, tower_grads):
        avg_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                grads.append(tf.expand_dims(g, 0))

            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_vars = (grad, v)
            avg_grads.append(grad_and_vars)

        return avg_grads