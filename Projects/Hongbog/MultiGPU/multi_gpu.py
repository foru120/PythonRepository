from Projects.Hongbog.MultiGPU.model import Model
from Projects.Hongbog.MultiGPU.constants import *

class MultiGPU:
    def __init__(self):
        self.model = Model()

    def init_tower(self, num_gpus, train_x, train_y, opt):
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (flags.FLAGS.tower_name, i)) as scope:
                        # x_batch, y_batch = batch_queue.dequeue()

                        loss = self._tower_loss(x_batch=train_x, y_batch=train_y, scope=scope)

                        tf.get_variable_scope().reuse_variables()

                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        grads = opt.compute_gradients(loss, colocate_gradients_with_ops=True)

                        tower_grads.append(grads)

        grads = self._avg_grads(tower_grads=tower_grads)

        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/grad', grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        return grads, loss, summaries

    def _tower_loss(self, x_batch, y_batch, scope):
        logits = self.model.build_graph(x_batch)

        tot_loss = self.model.loss(logits=logits, labels=y_batch, scope=scope)

        return tot_loss

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