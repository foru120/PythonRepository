import os
import tensorflow as tf
from Model import Nets

import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer:
    def __init__(self, data_loader, img_size, batch_size, n_epoch, learning_rate, drop_out_rate, decay_rate, model_save_path, n_class, act_func):
        self.data_loader = data_loader
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.drop_out_rate = drop_out_rate
        self.n_epoch = n_epoch
        self.decay_rate = decay_rate
        self.model_save_path = model_save_path + '/Focal_FCN.ckpt'
        self.net = Nets(data_shape=self.img_size, channel=1, n_class=n_class, act_func=act_func, batch_size=batch_size)
        print('>> Trainer Initialized')

    def _get_optimizer(self, global_step, num_lists):

        if num_lists * 2 > 2000:
            decay_step = num_lists * 2
        else:
            decay_step = 1000

        decay_rate = self.decay_rate

        learning_rate = self.learning_rate

        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=decay_step, decay_rate=decay_rate, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=exponential_decay_learning_rate).minimize(self.net.cost, global_step=global_step)

    def train_(self):
        num_train = self.data_loader.tr_count // self.batch_size + 1
        num_valid = self.data_loader.val_count // self.batch_size + 1
        
        print('>> num_train:', num_train)
        print('>> num_valid:', num_valid)
        global_step = tf.Variable(0, trainable=False)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._get_optimizer(global_step, num_train)

        train_input, train_label = self.data_loader.train_fn()
        valid_input, valid_label = self.data_loader.valid_fn()

        with tf.Session() as sess:
            print('>> Tensorflow session built. variables initialized')
            sess.run(tf.global_variables_initializer())
            # merge = tf.summary.merge_all()

            saver = tf.train.Saver()

            ckpt_st = tf.train.get_checkpoint_state(self.model_save_path)

            if ckpt_st is not None:
                saver.restore(sess, ckpt_st.model_checkpoint_path)

            tf.train.start_queue_runners(sess=sess)
            print('>> running started')
            for epoch in range(self.n_epoch):
                stt = time.time()
                total_loss, total_acc = 0., 0.

                for tstep in range(num_train):
                    st = time.time()
                    t_input, t_label = sess.run([train_input, train_label])
                    if tstep == 0:
                        print('>> t_input shape : ', t_input.shape, 't_label shape : ', t_label.shape)

                    feed_dict = {self.net.x: t_input,
                                 self.net.y: t_label,
                                 self.net.training: True,
                                 self.net.keep_prob: self.drop_out_rate}

                    loss, _ = sess.run([self.net.cost, self.optimizer], feed_dict=feed_dict)
                    et = time.time()
                    t = et - st
                    print(">> [Training] [%d/%d] step/e: %d  Loss: %.4f Step_Time: %.1f" % (epoch, self.n_epoch, tstep, loss, t))
                edt = time.time()
                for vstep in range(num_valid):
                    v_input, v_label = sess.run([valid_input, valid_label])
                    feed_dict = {self.net.x: v_input,
                                 self.net.y: v_label,
                                 self.net.training: False,
                                 self.net.keep_prob: self.drop_out_rate}

                    vloss, predictions, accuracy = sess.run([
                        self.net.cost, self.net.predict, self.net.accuracy], feed_dict=feed_dict)

                    total_loss += vloss
                    total_acc += accuracy

                total_loss /= num_valid
                total_acc /= num_valid

                print('>> [total/epoch] [%d/%d] / validation avg acc: %.4f / validation avg loss: %.4f / Training time(sec): %.2f' % (epoch, self.n_epoch, total_acc, total_loss, edt-stt))
                # train_writer = tf.summary.FileWriter('../log/test.summaries', sess.graph)

                saver.save(sess, self.model_save_path)
                print('>> model saved')

        sess.close()
        tf.reset_default_graph()
