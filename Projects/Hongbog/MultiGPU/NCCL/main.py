import os
import time
import numpy as np

from Projects.Hongbog.MultiGPU.NCCL.constants import *
from Projects.Hongbog.MultiGPU.NCCL.data_loader import DataLoader
from Projects.Hongbog.MultiGPU.NCCL.multi_gpu import MultiGPU
from tensorpack.graph_builder import override_to_local_variable

"""
    ▣ ShakeNet 신경망
"""
class Trainer:

    def __init__(self, seq, is_db_logging, is_ckpt_logging, is_cfm_logging, is_roc_logging, is_tb_logging, dataset, name):
        self.seq = seq
        self.best_epoch = -1
        self.is_db_logging = is_db_logging
        self.is_ckpt_logging = is_ckpt_logging
        self.is_cfm_logging = is_cfm_logging
        self.is_roc_logging = is_roc_logging
        self.is_tb_logging = is_tb_logging
        self.dataset = dataset
        self.name = name

        self._loader = DataLoader()
        print('>> The data loader has been initialized.')

        self._multi_gpu = MultiGPU()
        print('>> MultiGPU class has been initialized.')

    def get_post_init_ops(self):
        """
        Copy values of variables on GPU 0 to other GPUs.
        """
        # literally all variables, because it's better to sync optimizer-internal variables as well
        all_vars = tf.global_variables() + tf.local_variables()
        var_by_name = dict([(v.name, v) for v in all_vars])
        post_init_ops = []
        for v in all_vars:
            print(v.name)
            if not v.name.startswith('tower'):
                continue
            if v.name.startswith('tower0'):
                # no need for copy to tower0
                continue
            # in this trainer, the master name doesn't have the towerx/ prefix
            split_name = v.name.split('/')
            prefix = split_name[0]
            realname = '/'.join(split_name[1:])
            if prefix in realname:
                print("variable {} has its prefix {} appears multiple times in its name!".format(v.name, prefix))
            copy_from = var_by_name.get(v.name.replace(prefix, 'tower0'))
            if copy_from is not None:
                post_init_ops.append(v.assign(copy_from.read_value()))
            else:
                print("Cannot find {} in the graph!".format(realname))

        return tf.group(*post_init_ops, name='sync_variables_from_main_tower')

    def train(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

            train_step_num = (50000 // flags.FLAGS.batch_size // flags.FLAGS.num_gpus)

            # todo Data Loader Initialization
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0), trainable=False)

            decay_lr = tf.train.cosine_decay(flags.FLAGS.lr, global_step, train_step_num * flags.FLAGS.epochs)
            tf.summary.scalar('lr', decay_lr)

            opt = [tf.train.MomentumOptimizer(learning_rate=decay_lr, momentum=0.9, use_nesterov=True) for _ in range(flags.FLAGS.num_gpus)]

            train_x, train_y = self._loader.train_batch()

            batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                [train_x, train_y], capacity=2 * flags.FLAGS.num_gpus
            )

            acc, loss = self._multi_gpu.init_tower(num_gpus=flags.FLAGS.num_gpus,
                                                   batch_queue=batch_queue,
                                                   opt=opt)

            train_ops = []
            for idx, grads_and_vars in enumerate(self._multi_gpu.grads):
                with tf.name_scope('apply_gradients'), tf.device(tf.DeviceSpec(device_type='GPU', device_index=idx)):
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='tower%d' % idx)
                    with tf.control_dependencies(update_ops):
                        train_ops.append(opt[idx].apply_gradients(grads_and_vars, name='apply_grad_{}'.format(idx)))

            train_op = tf.group(*train_ops, name='train_op')

            saver = tf.train.Saver(tf.global_variables())

            # summary_op = tf.summary.merge(self._multi_gpu.summaries)
            summary_op = tf.summary.merge_all()

            self.sync_op = self.get_post_init_ops()

            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True,  # GPU 메모리 증분 할당
                                          per_process_gpu_memory_fraction=1.0),  # GPU 당 할당할 메모리 양
                allow_soft_placement=True,
                log_device_placement=False
            )

            tot_start_time = time.time()
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                sess.run(self.sync_op)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                summary_writer = tf.summary.FileWriter(flags.FLAGS.tensorboard_log_path, sess.graph)

                for step in range(flags.FLAGS.epochs * train_step_num):
                    step_start_time = time.time()
                    _, step_acc, step_loss = sess.run([train_op, acc, loss])
                    step_end_time = time.time()
                    print('■ Train - [Epoch-%d]/[Step-%d], acc: %.4f, loss: %.4f, time: %.3f'
                          % ((step // train_step_num) + 1, step % train_step_num, step_acc, step_loss, (step_end_time - step_start_time) / flags.FLAGS.num_gpus))

                    if step % train_step_num == 0:
                        sess.run(self.sync_op)

                    if step % 100 == 0 or (step + 1) == flags.FLAGS.epochs * train_step_num:
                        summary_writer.add_summary(sess.run(summary_op), step)

                    if step % 1000 == 0 or (step + 1) == flags.FLAGS.epochs * train_step_num:
                        saver.save(sess, os.path.join(flags.FLAGS.train_log_path, 'multigpu_model.ckpt'), global_step=step)

                coord.request_stop()
                coord.join(threads)
            tot_end_time = time.time()

            print('>>> Total Train Time: %.3f' % (tot_end_time - tot_start_time))

    def test(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

            test_step_num = (10000 // flags.FLAGS.batch_size)

            # todo Data Loader Initialization
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0), trainable=False)

            decay_lr = tf.train.cosine_decay(flags.FLAGS.lr, global_step, test_step_num * 1)

            opt = [tf.train.MomentumOptimizer(learning_rate=decay_lr, momentum=0.9, use_nesterov=True)]

            test_x, test_y = self._loader.test_batch()

            batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                [test_x, test_y], capacity=2 * flags.FLAGS.num_gpus
            )

            acc, loss = self._multi_gpu.init_tower(num_gpus=1,
                                                   batch_queue=batch_queue,
                                                   opt=opt)

            # self._multi_gpu.summaries.append(tf.summary.scalar('lr', decay_lr))

            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #
            # with tf.control_dependencies(update_ops):
            #     train_op = opt[0].apply_gradients(self._multi_gpu.grads, global_step=global_step)

            saver = tf.train.Saver(tf.global_variables())

            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True,  # GPU 메모리 증분 할당
                                          per_process_gpu_memory_fraction=1.0),  # GPU 당 할당할 메모리 양
                allow_soft_placement=True,
                log_device_placement=False
            )

            tot_start_time = time.time()
            with tf.Session(config=config) as sess:
                ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.train_log_path))

                if ckpt_st is not None:
                    '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                    saver.restore(sess, ckpt_st.model_checkpoint_path)
                    print('>> Model Restored')

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                tot_acc, tot_loss = [], []
                for step in range(test_step_num):
                    step_acc, step_loss = sess.run([acc, loss])
                    tot_acc.append(step_acc)
                    tot_loss.append(step_loss)
                tot_acc = np.asarray(tot_acc).mean()
                tot_loss = np.asarray(tot_loss).mean()

                print('■ Test - acc: %.4f, loss: %.4f' % (tot_acc, tot_loss))

                saver.save(sess, os.path.join(flags.FLAGS.deploy_log_path, 'multigpu_model.ckpt'), global_step=step)

                coord.request_stop()
                coord.join(threads)
            tot_end_time = time.time()

            print('>>> Total Test Time: %.3f' % (tot_end_time - tot_start_time))


trainer = Trainer(seq=4,
                  is_db_logging=True,
                  is_ckpt_logging=True,
                  is_cfm_logging=False,
                  is_roc_logging=False,
                  is_tb_logging=True,
                  dataset='cifar10',
                  name='shakenet-multigpu')
trainer.train()
# trainer.test()
# trainer.visualization(sample_per_class=5)