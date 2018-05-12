import time

import numpy as np
import tensorflow as tf
import os
from collections import deque
from PIL import Image

from DeepLearningTechniques.DC_GAN.dcgan_gogh import DCGAN
from DeepLearningTechniques.DC_GAN.database import Database
from DeepLearningTechniques.DC_GAN.dataloader import DataLoader

class Neuralnet:
    '''하이퍼파라미터 관련 변수'''
    _FLAGS = None


    '''훈련시 필요한 변수'''
    _model = None
    _saver = None
    _best_model_params = None

    '''모니터링 대상 변수'''
    train_gloss_mon = deque(maxlen=1000)  # deque: queue 의 양방향 삽입, 삭제, 검색이 가능
    train_dloss_mon = deque(maxlen=1000)

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
        flags.DEFINE_string('train_data_path', 'D:/Data/celeba/img/test', '학습 데이터 경로')
        flags.DEFINE_integer('epochs', 100, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 100, '훈련시 배치 크기')
        flags.DEFINE_integer('max_checks_without_progress', 20, '특정 횟수 만큼 조건이 만족하지 않은 경우(Early Stop Condition)')
        flags.DEFINE_string('trained_param_path',
                            'D:/Source/PythonRepository/DeepLearningTechniques/DC_GAN/train_log',
                            '훈련된 파라미터 값 저장 경로')

        self.config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1)
        )

    def create_image(self, images, epoch, sess):
        images = tf.image.convert_image_dtype(tf.transpose(tf.div(tf.add(images[np.random.permutation(self._FLAGS.batch_size)[:10]], 1.0), 2.0), perm=[0, 2, 1, 3]), tf.uint8)
        images = [image for image in tf.split(images, 10, axis=0)]

        for idx in range(0, len(images)):
            os.makedirs(os.path.join('gen_image', '1th_test', str(epoch)), exist_ok=True)
            with open('gen_image\\1th_test\\' + str(epoch) + '\\' + str(idx) + '.jpeg', mode='wb') as f:
                f.write(sess.run(tf.image.encode_jpeg(tf.squeeze(images[idx], [0]))))

    def train(self):
        num_train = self.loader.train_x_len // self._FLAGS.batch_size
        train_x = self.loader.train_loader()
        epoch = 1

        with tf.Session(config=self.config) as sess:
            dcgan = DCGAN(sess=sess, batch_size=self._FLAGS.batch_size)

            print('>> Tensorflow session built. Variables initialized')
            sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()

            ckpt_st = tf.train.get_checkpoint_state(self._FLAGS.trained_param_path)

            if ckpt_st is not None:
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('>> Running started')

            while True:
                tot_g_loss, tot_d_loss = 0., 0.
                sst = time.time()

                for step in range(num_train):
                    st = time.time()
                    batch_z = np.random.normal(size=(self._FLAGS.batch_size, 100))
                    # noise_z = np.random.normal(size=(self._FLAGS.batch_size, 96, 128, 32))
                    train_data = sess.run(train_x)

                    step_d_loss = 0.
                    for _ in range(1):
                        d_loss, _ = dcgan.d_train(batch_z, train_data)
                        tot_d_loss += d_loss
                        step_d_loss += d_loss

                    step_g_loss = 0.
                    for _ in range(2):
                        g_loss, _ = dcgan.g_train(batch_z)
                        tot_g_loss += g_loss
                        step_g_loss += g_loss

                    step_g_loss = step_g_loss / 2
                    step_d_loss = step_d_loss / 1
                    et = time.time()

                    print(">> [Training] epoch/step: [%d/%d], g_loss: %.6f, d_loss: %.6f, Step_Time: %.2f" % (
                        epoch, step, step_g_loss, step_d_loss, et-st))

                eet = time.time()
                self.db.mon_data_to_db(epoch, tot_g_loss, tot_d_loss, eet-sst)

                if epoch % 10 == 0:
                    os.makedirs(os.path.join('train_log', str(epoch).zfill(6)), exist_ok=True)
                    self._saver.save(sess, os.path.join(self._FLAGS.trained_param_path, str(epoch).zfill(6), 'image_processing_param.ckpt'))
                    img = dcgan.generate(np.random.normal(size=(self._FLAGS.batch_size, 100))) # , np.random.randn(self._FLAGS.batch_size, 96, 128, 32)
                    self.create_image(img, epoch, sess)
                    print('>> [Model & Image Saved] epoch: %d' % (epoch))

                epoch += 1

            coord.request_stop()
            coord.join(threads)

            self.db.close_conn()

neuralnet = Neuralnet(is_train=True, save_type='db')
neuralnet.train()