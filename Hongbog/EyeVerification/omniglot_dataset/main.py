import time
from collections import Counter
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re

from Hongbog.EyeVerification.omniglot_dataset.one_shot_model import Model
from Hongbog.EyeVerification.omniglot_dataset.dataloader import DataLoader

class Neuralnet:
    '''하이퍼파라미터 관련 변수'''
    _FLAGS = None

    '''훈련시 필요한 변수'''
    _model = None
    _saver = None
    _best_model_params = None

    def __init__(self, is_train, save_type=None):
        self.is_train = is_train
        self.save_type = save_type
        self._flag_setting()  # flag setting

        if (self.is_train == True) and (save_type == 'db'):  # data loading
            # self.db = Database(FLAGS=self._FLAGS, train_log=1)
            # self.db.init_database()
            self.train_loader = DataLoader(data_root_path=self._FLAGS.data_root_path, batch_size=self._FLAGS.batch_size,
                                           n_way=self._FLAGS.n_way, k_shot=self._FLAGS.k_shot, train_mode=True)
            self.eval_loader = DataLoader(data_root_path=self._FLAGS.data_root_path, batch_size=self._FLAGS.batch_size,
                                          n_way=self._FLAGS.n_way, k_shot=self._FLAGS.k_shot, train_mode=False)

    def _flag_setting(self):
        '''
        하이퍼파라미터 값을 설정하는 함수
        :return: None
        '''
        flags = tf.app.flags
        self._FLAGS = flags.FLAGS
        flags.DEFINE_string('train_right_root_path', 'D:\\100_dataset\\eye_verification\\eye_only_v2\\train\\right', '오른쪽 눈 학습 데이터 경로')
        flags.DEFINE_string('test_right_root_path', 'D:\\100_dataset\\eye_verification\\eye_only_v2\\test\\right', '오른쪽 눈 테스트 데이터 경로')
        flags.DEFINE_string('train_left_root_path', 'D:\\100_dataset\\eye_verification\\eye_only_v2\\train\\left', '왼쪽 눈 학습 데이터 경로')
        flags.DEFINE_string('test_left_root_path', 'D:\\100_dataset\\eye_verification\\eye_only_v2\\test\\left', '왼쪽 눈 테스트 데이터 경로')
        flags.DEFINE_string('data_root_path', 'D:\\100_dataset\\omniglot', '데이터 루트 경로')
        flags.DEFINE_integer('epochs', 200, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 50, '훈련시 배치 크기')
        flags.DEFINE_integer('n_way', 20, 'One-Shot Learning class 개수')
        flags.DEFINE_integer('k_shot', 1, 'One-shot Learning sample 개수')
        flags.DEFINE_float('lr', 1e-3, 'learning rate')
        flags.DEFINE_integer('max_checks_without_progress', 20, '특정 횟수 만큼 조건이 만족하지 않은 경우(Early Stop Condition)')
        flags.DEFINE_string('trained_param_path',
                            'D:\\05_source\\PythonRepository\\Hongbog\\EyeVerification\\omniglot_dataset\\train_log\\1th_test',
                            '훈련된 파라미터 값 저장 경로')
        self.config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1)
        )

    def train(self):
        with tf.Session(config=self.config) as sess:
            model = Model(sess=sess, name='one-shot', batch_size=self._FLAGS.batch_size, n_way=self._FLAGS.n_way,
                          k_shot=self._FLAGS.k_shot, training=True, use_fce=True, lr=self._FLAGS.lr)

            print('>> Tensorflow session built. Variables initialized.')
            sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver(max_to_keep=10)

            # ckpt_st = tf.train.get_checkpoint_state(os.path.join(self._FLAGS.trained_param_path, '000001'))

            # if ckpt_st is not None:
            #     self._saver.restore(sess, ckpt_st.model_checkpoint_path)
            #     print('>> Model Restored')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('>> Running started.')

            for epoch in range(1, self._FLAGS.epochs+1):
                '''Model Train'''
                correct = []
                train_st = time.time()
                for step in range(1, self.train_loader.iters+1):
                    x_set, y_set, x_hat, y_hat = self.train_loader.next_batch()
                    logits, prediction, loss, _ = model.train(x_set, y_set, x_hat, y_hat)
                    correct.append(np.equal(prediction, y_hat))

                    if step % 100 == 0:
                        print('>> [Training-Step] epoch: %3d, step: %3d, loss: %.3f, acc: %.2f' % (epoch, step, loss, np.mean(np.equal(prediction, y_hat))*100))
                train_et = time.time()
                print('>> [Training-Total] acc: %.2f, time: %.2f' % (np.mean(np.stack(correct)) * 100, train_et-train_st))

                '''Model Test'''
                correct = []
                for step in range(1, self.eval_loader.iters+1):
                    x_set, y_set, x_hat, y_hat = self.eval_loader.next_batch()
                    logits, prediction = model.test(x_set, y_set, x_hat)
                    correct.append(np.equal(prediction, y_hat))
                print('>> [Evaluation] acc: %.2f' % (np.mean(np.stack(correct)) * 100))

                ## Save model
                os.makedirs(os.path.join(self._FLAGS.trained_param_path), exist_ok=True)
                self._saver.save(sess, os.path.join(self._FLAGS.trained_param_path, 'one_shot_param'), global_step=epoch)
                print('>> [Model saved] epoch: %d' % (epoch))

            coord.request_stop()
            coord.join(threads)

            # self.db.close_conn()

    def cam_test(self):
        sample_num = 3  # 클래스 당 테스트 샘플 개수
        class_num = 7  # 전체 클래스 개수
        batch_size = 3 * 7
        img_size = (100, 50)
        right_sample_path = 'D:\\100_dataset\\eye_verification\\eye_only_v2\\test\\right'
        left_sample_path = 'D:\\100_dataset\\eye_verification\\eye_only_v2\\test\\left'

        def save_matplot_img(right_outputs, left_outputs, sample_num, class_num):
            f = plt.figure(figsize=(10, 8))
            outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
            right_inner = gridspec.GridSpecFromSubplotSpec(7, 3, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
            left_inner = gridspec.GridSpecFromSubplotSpec(7, 3, subplot_spec=outer[1], wspace=0.1, hspace=0.1)

            for cls in range(class_num):
                for sample in range(sample_num):
                    subplot = plt.Subplot(f, right_inner[sample + cls * sample_num])
                    subplot.imshow(right_outputs[sample + cls * sample_num])
                    f.add_subplot(subplot)

                    subplot = plt.Subplot(f, left_inner[sample + cls * sample_num])
                    subplot.imshow(left_outputs[sample + cls * sample_num])
                    f.add_subplot(subplot)

            f.savefig('D:\\grad_cam\\cam_test.png')
            print('>> Grad CAM Complete')

        def get_file_names():
            right_file_names, left_file_names = [], []

            for cls in range(class_num):
                right_file_name = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(os.path.join(right_sample_path, str(cls)))]
                right_file_name = np.array(right_file_name).flatten()

                left_file_name = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(os.path.join(left_sample_path, str(cls)))]
                left_file_name = np.array(left_file_name).flatten()

                random_sort = np.random.permutation(right_file_name.shape[0])
                right_file_name = right_file_name[random_sort][:sample_num]
                left_file_name = left_file_name[random_sort][:sample_num]

                for r_file_name, l_file_name in zip(right_file_name, left_file_name):
                    right_file_names.append(r_file_name)
                    left_file_names.append(l_file_name)

            right_file_names = tf.convert_to_tensor(right_file_names, dtype=tf.string)
            left_file_names = tf.convert_to_tensor(left_file_names, dtype=tf.string)

            return right_file_names, left_file_names

        def data_preprocessing(path):
            with tf.variable_scope('data_preprocessing'):
                data = tf.read_file(path)
                data = tf.image.decode_png(data, channels=1, name='decode_img')
                data = tf.image.resize_images(data, size=img_size)
                data = tf.divide(data, 255.)
            return data

        def data_loader(right_file_names, left_file_names):
            with tf.variable_scope('data_loader'):
                # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
                right_dataset = tf.contrib.data.Dataset.from_tensor_slices(right_file_names).repeat()
                left_dataset = tf.contrib.data.Dataset.from_tensor_slices(left_file_names).repeat()

                right_normal_dataset_map = right_dataset.map(data_preprocessing).batch(batch_size)
                right_normal_iterator = right_normal_dataset_map.make_one_shot_iterator()
                right_normal_batch_input = right_normal_iterator.get_next()

                left_normal_dataset_map = left_dataset.map(data_preprocessing).batch(batch_size)
                left_normal_iterator = left_normal_dataset_map.make_one_shot_iterator()
                left_normal_batch_input = left_normal_iterator.get_next()

            return right_normal_batch_input, left_normal_batch_input

        right_file_names, left_file_names = get_file_names()
        right_batch_img, left_batch_img = data_loader(right_file_names, left_file_names)

        with tf.Session() as sess:
            right_model = Model(sess=sess, name='right', training=self.is_train)
            left_model = Model(sess=sess, name='left', training=self.is_train)

            self._saver = tf.train.Saver()
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(self._FLAGS.trained_param_path))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            right_batch_x, left_batch_x = sess.run([right_batch_img, left_batch_img])
            right_file_names, left_file_names = sess.run([right_file_names, left_file_names])

            right_grad_cam = right_model.grad_cam(right_batch_x, sample_num * class_num)
            left_grad_cam = left_model.grad_cam(left_batch_x, sample_num * class_num)

            # 원본 영상과 Grad CAM 을 합친 결과
            right_outputs = []
            left_outputs = []

            for idx in range(sample_num * class_num):
                # 오른쪽 눈
                right_img = Image.open(right_file_names[idx], mode='r').convert('RGB')
                right_img = np.array(right_img.resize(img_size))
                right_img = right_img.astype(float)
                right_img /= 255.

                right_cam = cv2.applyColorMap(np.uint8(255 * right_grad_cam[idx]), cv2.COLORMAP_JET)
                right_cam = cv2.cvtColor(right_cam, cv2.COLOR_BGR2RGB)

                # grad-cam과 원본 이미지 중첩.
                alpha = 0.0025  # COLORMAP_JET 의 비율 값.
                right_output = right_img + alpha * right_cam
                right_output /= right_output.max()

                # 왼쪽 눈
                left_img = Image.open(left_file_names[idx], mode='r').convert('RGB')
                left_img = np.array(left_img.resize(img_size))
                left_img = left_img.astype(float)
                left_img /= 255.

                left_cam = cv2.applyColorMap(np.uint8(255 * left_grad_cam[idx]), cv2.COLORMAP_JET)
                left_cam = cv2.cvtColor(left_cam, cv2.COLOR_BGR2RGB)

                # grad-cam과 원본 이미지 중첩.
                alpha = 0.0025  # COLORMAP_JET 의 비율 값.
                left_output = left_img + alpha * left_cam
                left_output /= left_output.max()

                right_outputs.append(right_output)
                left_outputs.append(left_output)

            save_matplot_img(right_outputs, left_outputs, sample_num, class_num)

    def test(self):
        batch_size = 50
        img_size = (100, 50)
        right_sample_path = 'D:\\100_dataset\\eye_verification\\eye_only_v3\\test\\right\\0'
        left_sample_path = 'D:\\100_dataset\\eye_verification\\eye_only_v3\\test\\left\\0'

        def get_file_names():
            right_file_names, left_file_names = [], []

            right_file_name = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(right_sample_path)]
            right_file_name = np.array(right_file_name).flatten()

            left_file_name = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(left_sample_path)]
            left_file_name = np.array(left_file_name).flatten()

            random_sort = np.random.permutation(right_file_name.shape[0])
            right_file_name = right_file_name[random_sort]
            left_file_name = left_file_name[random_sort]

            for r_file_name, l_file_name in zip(right_file_name, left_file_name):
                right_file_names.append(r_file_name)
                left_file_names.append(l_file_name)

            right_file_names = tf.convert_to_tensor(right_file_names, dtype=tf.string)
            left_file_names = tf.convert_to_tensor(left_file_names, dtype=tf.string)

            return right_file_names, left_file_names

        def data_preprocessing(path):
            with tf.variable_scope('data_preprocessing'):
                data = tf.read_file(path)
                data = tf.image.decode_png(data, channels=1, name='decode_img')
                data = tf.image.resize_images(data, size=img_size)
                data = tf.divide(data, 255.)
            return data

        def data_loader(right_file_names, left_file_names):
            with tf.variable_scope('data_loader'):
                # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
                right_dataset = tf.contrib.data.Dataset.from_tensor_slices(right_file_names).repeat()
                left_dataset = tf.contrib.data.Dataset.from_tensor_slices(left_file_names).repeat()

                right_normal_dataset_map = right_dataset.map(data_preprocessing).batch(batch_size)
                right_normal_iterator = right_normal_dataset_map.make_one_shot_iterator()
                right_normal_batch_input = right_normal_iterator.get_next()

                left_normal_dataset_map = left_dataset.map(data_preprocessing).batch(batch_size)
                left_normal_iterator = left_normal_dataset_map.make_one_shot_iterator()
                left_normal_batch_input = left_normal_iterator.get_next()

            return right_normal_batch_input, left_normal_batch_input

        right_file_names, left_file_names = get_file_names()
        right_batch_img, left_batch_img = data_loader(right_file_names, left_file_names)

        with tf.Session() as sess:
            right_model = Model(sess=sess, name='right', training=self.is_train)
            left_model = Model(sess=sess, name='left', training=self.is_train)

            self._saver = tf.train.Saver()
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(self._FLAGS.trained_param_path))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                print(ckpt_st.model_checkpoint_path)
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'right/output_layer/logit/W_conv2d')[0]
            # print(sess.run(kernel))

            ensemble_pred = []
            for _ in range(0, right_file_names.shape[0], batch_size):
                right_batch_x, left_batch_x = sess.run([right_batch_img, left_batch_img])
                right_prob = right_model.predict(x_test=right_batch_x)
                left_prob = left_model.predict(x_test=left_batch_x)

                ensemble_pred.append(np.argmax(right_prob + left_prob, axis=1).tolist())

            '''Ensemble Prediction'''
            ensemble_pred = np.array(ensemble_pred).flatten()
            print(ensemble_pred)
            ensemble_pred = Counter(ensemble_pred)
            print(ensemble_pred.most_common(1)[0][0])

neuralnet = Neuralnet(is_train=True, save_type='db')
neuralnet.train()
# neuralnet.cam_test()
# neuralnet.test()