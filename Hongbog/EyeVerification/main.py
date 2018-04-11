import time

import numpy as np
import tensorflow as tf
import os

from Hongbog.EyeVerification.model import Model
from Hongbog.EyeVerification.database import Database
from Hongbog.EyeVerification.dataloader import DataLoader

class Neuralnet:
    '''하이퍼파라미터 관련 변수'''
    _FLAGS = None

    '''훈련시 필요한 변수'''
    _model = None
    _saver = None
    _best_model_params = None

    def __init__(self, is_train, save_type=None):
        self.save_type = save_type
        self._flag_setting()  # flag setting

        if (is_train == True) and (save_type == 'db'):  # data loading
            self.db = Database(FLAGS=self._FLAGS, train_log=1)
            self.db.init_database()
            self.loader = DataLoader(batch_size=self._FLAGS.batch_size, train_data_path=self._FLAGS.train_data_path)

    def _flag_setting(self):
        '''
        하이퍼파라미터 값을 설정하는 함수
        :return: None
        '''
        flags = tf.app.flags
        self._FLAGS = flags.FLAGS
        flags.DEFINE_string('train_data_path', 'D:/Data/eye_verification/train', '학습 데이터 경로')
        flags.DEFINE_string('test_data_path', 'D:/Data/eye_verification/test', '테스트 데이터 경로')
        flags.DEFINE_integer('epochs', 100, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 100, '훈련시 배치 크기')
        flags.DEFINE_integer('max_checks_without_progress', 20, '특정 횟수 만큼 조건이 만족하지 않은 경우(Early Stop Condition)')
        flags.DEFINE_string('trained_param_path',
                            'D:/Source/PythonRepository/DeepLearningTechniques/GAN/BEGAN/train_log/1th_test',
                            '훈련된 파라미터 값 저장 경로')
        self.config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1)
        )

    def train(self):
        train_num = self.loader.train_x_len // self._FLAGS.batch_size
        test_num = self.loader.test_x_len // self._FLAGS.batch_size
        train_x, train_y = self.loader.train_loader()
        test_x, test_y = self.loader.train_loader()

        epoch = 1

        with tf.Session(config=self.config) as sess:
            model = Model(sess=sess)

            print('>> Tensorflow session built. Variables initialized.')
            sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()

            # ckpt_st = tf.train.get_checkpoint_state(os.path.join(self._FLAGS.trained_param_path, '000001'))

            # if ckpt_st is not None:
            #     self._saver.restore(sess, ckpt_st.model_checkpoint_path)
            #     print('>> Model Restored')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('>> Running started.')

            for epoch in range(1, self._FLAGS.epochs+1):
                tot_train_acc, tot_train_loss = [], []
                tot_test_acc, tot_test_loss = [],[]

                train_st = time.time()
                for step in range(1, train_num+1):
                    st = time.time()
                    train_batch_x, train_batch_y = sess.run([train_x, train_y])
                    train_acc, train_loss, _ = model.train(train_batch_x, train_batch_y)

                    tot_train_acc.append(train_acc)
                    tot_train_loss.append(train_loss)
                    et = time.time()
                    print(">> [Training] epoch/step: [%d/%d], Accuracy: %.6f, Loss: %.6f, step_time: %.2f" % (epoch, step, train_acc, train_loss, et - st))
                train_et = time.time()

                tot_train_time = train_et - train_st

                test_st = time.time()
                for step in range(1, test_num + 1):
                    st = time.time()
                    test_batch_x, test_batch_y = sess.run([test_x, test_y])
                    test_acc, test_loss, _ = model.train(test_batch_x, test_batch_y)

                    tot_test_acc.append(test_acc)
                    tot_test_loss.append(test_loss)
                    et = time.time()
                    print(">> [Test] epoch/step: [%d/%d], Accuracy: %.6f, Loss: %.6f, step_time: %.2f" % (epoch, step, test_acc, test_loss, et - st))
                test_et = time.time()

                tot_test_time = test_et - test_st

                tot_train_acc = np.mean(np.array(tot_train_acc))
                tot_train_loss = np.mean(np.array(tot_train_loss))
                tot_test_acc = np.mean(np.array(tot_test_acc))
                tot_test_loss = np.mean(np.array(tot_test_loss))

                self.db.mon_data_to_db(epoch, tot_train_acc, tot_test_acc, tot_train_loss, tot_test_loss, tot_train_time, tot_test_time)

                ## Save model
                os.makedirs(os.path.join('train_log/1th_test', str(epoch).zfill(6)), exist_ok=True)
                self._saver.save(sess, os.path.join(self._FLAGS.trained_param_path, str(epoch).zfill(6), 'image_processing_param.ckpt'))
                print('>> [Model saved] epoch: %d' % (epoch))

            coord.request_stop()
            coord.join(threads)

            self.db.close_conn()

neuralnet = Neuralnet(is_train=True, save_type='db')
neuralnet.train()