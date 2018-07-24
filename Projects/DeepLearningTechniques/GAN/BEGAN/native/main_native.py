import time

import numpy as np
import tensorflow as tf
import os

from DeepLearningTechniques.GAN.BEGAN.slim.model_slim import BEGAN
# from DeepLearningTechniques.GAN.BEGAN.database import Database
from DeepLearningTechniques.GAN.BEGAN.native.dataloader import DataLoader

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
            # self.db = Database(FLAGS=self._FLAGS, train_log=18)
            # self.db.init_database()
            self.loader = DataLoader(batch_size=self._FLAGS.batch_size, train_data_path=self._FLAGS.train_data_path)

    def _flag_setting(self):
        '''
        하이퍼파라미터 값을 설정하는 함수
        :return: None
        '''
        flags = tf.app.flags
        self._FLAGS = flags.FLAGS
        flags.DEFINE_string('train_data_path', '/home/kyh/dataset/celeba/img/img_align_celeba_png', '학습 데이터 경로')
        flags.DEFINE_integer('epochs', 100, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 16, '훈련시 배치 크기')
        flags.DEFINE_integer('max_checks_without_progress', 20, '특정 횟수 만큼 조건이 만족하지 않은 경우 (Early Stop Condition)')
        flags.DEFINE_string('trained_param_path',
                            '/home/kyh/PycharmProjects/PythonRepository/DeepLearningTechniques/GAN/BEGAN/train_log/19th_test',
                            '훈련된 파라미터 값 저장 경로')
        self.config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1)
        )

    def create_image(self, images, epoch, step, start_idx, sess):
        images = tf.transpose(images, perm=[0, 2, 1, 3])
        images = tf.image.convert_image_dtype(images / 255., tf.uint8)
        images = [image for image in tf.split(images, self._FLAGS.batch_size, axis=0)]

        for idx in range(0, len(images)):
            os.makedirs(os.path.join('gen_image', '19th_test', str(epoch).zfill(6) + '_' + str(step).zfill(6)), exist_ok=True)
            with open(os.path.join('gen_image', '19th_test', str(epoch).zfill(6) + '_' + str(step).zfill(6), str(idx + start_idx) + '.jpeg'), mode='wb') as f:
                f.write(sess.run(tf.image.encode_jpeg(tf.squeeze(images[idx], [0]))))

    def train(self):
        num_train = self.loader.train_x_len // self._FLAGS.batch_size
        print('>> Dataset-CelebA, Total CNT:', self.loader.train_x_len, ', Step per Epoch:', num_train)
        train_x = self.loader.train_loader()
        epoch = 1
        global_step = 1

        with tf.Session(config=self.config) as sess:
            began = BEGAN(sess=sess, batch_size=self._FLAGS.batch_size)

            print('>> Tensorflow session built. Variables initialized.')
            sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver(max_to_keep=20)

            # ckpt_st = tf.train.get_checkpoint_state(os.path.join(self._FLAGS.trained_param_path, '000001'))

            # if ckpt_st is not None:
            #     self._saver.restore(sess, ckpt_st.model_checkpoint_path)
            #     print('>> Model Restored')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('>> Running started.')

            while True:
                for step in range(1, num_train+1):
                    st = time.time()
                    z = np.random.uniform(-1, 1, size=(self._FLAGS.batch_size, 64))
                    x = sess.run(train_x)

                    if global_step % 10000 == 0:
                        began.learning_rate = max(0.00002, began.learning_rate/2)

                    d_loss, g_loss, k_t, measure, _ = began.train(z=z, x=x)

                    et = time.time()

                    print(">> [Training] epoch/step/global_step: [%d/%d/%d], g_loss: %.6f, d_loss: %.6f, k_t: %.4f, measure: %.4f, step_time: %.2f" % (
                        epoch, step, global_step, g_loss, d_loss, k_t, measure, et-st))

                    # self.db.mon_data_to_db(epoch, step, float(g_loss), float(d_loss), float(k_t), float(measure), et-st)

                    if global_step % 2000 == 0:
                        ## Save Model & image
                        os.makedirs(os.path.join(self._FLAGS.trained_param_path), exist_ok=True)
                        self._saver.save(sess, os.path.join(self._FLAGS.trained_param_path, 'began_param'), global_step=global_step)
                        for idx in range(5):
                            g_img = began.generate(z=np.random.uniform(-1, 1, size=(self._FLAGS.batch_size, 64)))
                            self.create_image(g_img, epoch, step, idx * self._FLAGS.batch_size, sess)
                        print('>> [Model & Image Saved] epoch: %d, step: %d' % (epoch, step))
                    global_step += 1

                epoch += 1

            coord.request_stop()
            coord.join(threads)

            # self.db.close_conn()

neuralnet = Neuralnet(is_train=True, save_type='db')
neuralnet.train()