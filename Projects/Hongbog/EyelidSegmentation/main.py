import tensorflow as tf
import numpy as np
import os
import time
from collections import deque
import datetime

from PIL import Image
from PIL import ImageGrab
import matplotlib.pyplot as plt

from Hongbog.EyelidSegmentation.model_resnet_v1 import Model
# from Hongbog.EyelidSegmentation.model import Model
from Hongbog.EyelidSegmentation.database import Database

class Neuralnet:
    '''신경망을 훈련하기 위한 클래스'''

    '''데이터 관련 변수'''
    _FLAGS = None

    '''데이터 관련 변수'''
    _tot_x, _tot_y = None, None
    _train_x, _train_y = None, None
    _test_x, _test_y = None, None
    _valid_x, _valid_y = None, None

    '''훈련시 필요한 변수'''
    _model = None
    _saver = None
    _best_model_params = None

    '''모니터링 대상 변수'''
    train_acc_mon = deque(maxlen=100)
    valid_acc_mon = deque(maxlen=100)
    train_loss_mon = deque(maxlen=100)
    valid_loss_mon = deque(maxlen=100)

    def __init__(self, is_train, save_type=None):
        self.save_type = save_type
        self._flag_setting()  # flag setting

        if is_train == True:  # data loading
            print('Data load start!!!')
            stime = time.time()
            self._data_loading()
            self._data_separation()
            etime = time.time()
            print('Data loaded!!! - ' + str(round(etime - stime, 2)) + ' 초.')

            if save_type == 'db':
                self.db = Database(self._FLAGS, 16)
                self.db.init_database()

    def _flag_setting(self):
        '''
        하이퍼파라미터 값을 설정하는 함수
        :return: None
        '''
        flags = tf.app.flags
        self._FLAGS = flags.FLAGS
        flags.DEFINE_string('image_data_path', 'D:\\100_dataset\\casia_eyelid_segmentation\\image_data', '훈련 이미지 데이터 경로')
        flags.DEFINE_integer('epochs', 300, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 100, '훈련시 배치 크기')
        flags.DEFINE_integer('max_checks_without_progress', 100, '특정 횟수 만큼 조건이 만족하지 않은 경우')
        flags.DEFINE_string('trained_param_path', 'D:/05_source/PythonRepository/Hongbog/EyelidSegmentation/train_log/0016/image_processing_param.ckpt', '훈련된 파라미터 값 저장 경로')
        flags.DEFINE_string('mon_data_log_path', 'D:/05_source/PythonRepository/Hongbog/EyelidSegmentation/mon_log/mon_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.txt', '훈련시 모니터링 데이터 저장 경로')
        self.config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
        )

    def _data_loading(self):
        '''
        최종 전처리된 데이터 값을 로딩하는 함수
        :return: None
        '''
        tot_data = None

        for idx, file_name in enumerate(os.listdir(self._FLAGS.image_data_path)):
            data = np.loadtxt(os.path.join(self._FLAGS.image_data_path, file_name), delimiter=',')
            if idx == 0:
                tot_data = data
            else:
                tot_data = np.concatenate((tot_data, data), axis=0)
        np.random.shuffle(tot_data)
        self._tot_x, self._tot_y = tot_data[:, 0:-1] / 255, tot_data[:, -1]

    def _data_separation(self):
        '''
        로딩된 전체 데이터에 대해 훈련 데이터, 테스트 데이터, 검증 데이터로 분리하는 함수 (6:3:1 비율)
        :return: None
        '''
        train_end_idx, test_end_idx, valid_end_idx = int(self._tot_x.shape[0] * 0.6), int(self._tot_x.shape[0] * 0.9), int(self._tot_x.shape[0])
        self._train_x, self._train_y = self._tot_x[0:train_end_idx, ], self._tot_y[0:train_end_idx, ]
        self._test_x,  self._test_y  = self._tot_x[train_end_idx:test_end_idx, ], self._tot_y[train_end_idx:test_end_idx, ]
        self._valid_x, self._valid_y = self._tot_x[test_end_idx:, ], self._tot_y[test_end_idx:, ]
        print('train :', str(len(self._train_x)), '개 , test :', str(len(self._test_x)), '개 , validation :', str(len(self._valid_x)), '개')

    def _get_model_params(self):
        '''
        텐서플로우 내에 존재하는 모든 파라미터들의 값을 추출하는 함수
        :return: 텐서플로우 내의 {변수: 값}, type -> Dict
        '''
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

    def _restore_model_params(self):
        '''
        메모리에 존재하는 파라미터들의 값을 Restore 하는 함수
        :return: None
        '''
        gvar_names = list(self._best_model_params.keys())
        assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}  # inputs : 해당 operation 의 입력 데이터를 표현하는 objects
        feed_dict = {init_values[gvar_name]: self._best_model_params[gvar_name] for gvar_name in gvar_names}
        tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

    def _image_screeshot(self):
        '''
        이미지 캡쳐하는 함수
        :return: None
        '''
        im = ImageGrab.grab()
        im.show()

    def train(self):
        with tf.Session(config=self.config) as sess:
            self._model = Model(sess)
            sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()
            best_loss_val = np.infty  # 가장 좋은 loss 값을 저장하는 변수
            check_since_last_progress = 0  # early stopping 조건을 만족하지 않은 횟수
            self._best_model_params = None  # 가장 좋은 모델의 parameter 값을 저장하는 변수

            print('Train start!!')

            for epoch in range(self._FLAGS.epochs):
                train_acc_list = []
                valid_acc_list = []
                tot_train_loss = 0.
                tot_valid_loss = 0.

                stime = time.time()

                # 훈련 부분
                for idx in range(0, self._train_x.shape[0], self._FLAGS.batch_size):
                    train_x_batch, train_y_batch = self._train_x[idx:idx+self._FLAGS.batch_size, ], self._train_y[idx:idx+self._FLAGS.batch_size, ]
                    if epoch+1 in (50, 150):  # dynamic learning rate
                        self._model.learning_rate = self._model.learning_rate/10
                    train_acc, train_loss, _ = self._model.train(train_x_batch.reshape(-1, 16, 16, 1), train_y_batch)
                    tot_train_loss += train_loss / len(train_x_batch)
                    train_acc_list.append(train_acc)
                # 검증 부분
                for idx in range(0, self._valid_x.shape[0], self._FLAGS.batch_size):
                    valid_x_batch, valid_y_batch = self._valid_x[idx:idx+self._FLAGS.batch_size, ], self._valid_y[idx:idx+self._FLAGS.batch_size, ]
                    valid_acc, valid_loss = self._model.validation(valid_x_batch.reshape(-1, 16, 16, 1), valid_y_batch)
                    tot_valid_loss += valid_loss / len(valid_x_batch)
                    valid_acc_list.append(valid_acc)

                etime = time.time()

                train_acc_mon = round(np.mean(np.array(train_acc_list)), 4)
                train_loss_mon = round(tot_train_loss, 4)
                valid_acc_mon = round(np.mean(np.array(valid_acc_list)), 4)
                valid_loss_mon = round(tot_valid_loss, 4)
                train_time = round(etime - stime, 2)

                self.train_acc_mon.append(train_acc_mon)
                self.train_loss_mon.append(train_loss_mon)
                self.valid_acc_mon.append(valid_acc_mon)
                self.valid_loss_mon.append(valid_loss_mon)

                print('epoch:', epoch + 1, ', train accuracy:', train_acc_mon, ', train loss:', train_loss_mon,
                      ', validation accuracy:', valid_acc_mon, ', validation loss:', valid_loss_mon, ', train time:',
                      train_time)

                if self.save_type == 'file':
                    self.db.mon_data_to_file(train_acc_mon, train_loss_mon, valid_acc_mon, valid_loss_mon)
                else:
                    self.db.mon_data_to_db(train_acc_mon, train_loss_mon, valid_acc_mon, valid_loss_mon, train_time)

                # Early Stopping 조건 확인
                if tot_valid_loss < best_loss_val:
                    best_loss_val = tot_valid_loss
                    check_since_last_progress = 0
                    self._best_model_params = self._get_model_params()
                    self._saver.save(sess, self._FLAGS.trained_param_path)
                else:
                    check_since_last_progress += 1

                # self._FLAGS.max_checks_without_progress 에 설정된 값(횟수) 만큼 학습의 최적 loss 값이 변경되지 않을 경우(횟수) Early Stopping 수행
                if check_since_last_progress > self._FLAGS.max_checks_without_progress:
                    print('Early stopping - epoch(' + str(epoch+1) + ')')
                    break

            self._test()

    def _test(self):
        '''
        테스트 데이터에 대해 분류를 수행하는 함수
        :return: None
        '''
        print('Test start!!!')
        if self._best_model_params:  # 가장 좋은 신경망의 파라미터 값을 Restore
            self._restore_model_params()

        test_acc_list = []

        for idx in range(0, self._test_x.shape[0], self._FLAGS.batch_size):
            test_x_batch, test_y_batch = self._test_x[idx:idx+self._FLAGS.batch_size, ], self._test_y[idx:idx+self._FLAGS.batch_size, ]
            a = self._model.get_accuracy(test_x_batch.reshape(-1, 16, 16, 1), test_y_batch)
            test_acc_list.append(a)

        print('test accuracy:', np.mean(np.array(test_acc_list)))

        self.db.close_conn()

    def _create_patch_image(self, ori_img):
        '''
        패치 사이즈 단위로 이미지 잘라내는 함수
        :param ori_img: 원본 이미지
        :return: 패치 단위로 분리된 이미지 데이터, type -> ndarray
        '''
        x_pixel, y_pixel = ori_img.size
        x_delta, y_delta = int(x_pixel/40), int(y_pixel/30)
        patches_data = []

        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                img_data = []
                for y in range(init_y, init_y + y_delta):
                    for x in range(init_x, init_x + x_delta):
                        img_data.append(ori_img.getpixel((x, y)))
                patches_data.append(img_data)
        return np.array(patches_data)

    def _edge_output_on_the_image(self, ori_img, predict_edges):
        '''
        예측된 edge 들을 이미지 상에 표시하기 위한 함수
        :param ori_img: 원본 이미지
        :param predict_edges: 예측된 edge 들
        :return: edge 가 표시된 이미지, type -> Image
        '''
        x_pixel, y_pixel = ori_img.size
        x_delta, y_delta, img_cnt = int(x_pixel / 40), int(y_pixel / 30), 1
        rgb_img = ori_img.convert('RGB')

        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                if img_cnt in predict_edges:
                    for y in range(init_y, init_y + y_delta):
                        for x in range(init_x, init_x + x_delta):
                            rgb_img.putpixel((x, y), (157, 195, 230))
                img_cnt = img_cnt + 1

        return rgb_img

    def predict(self, img_path):
        '''
        임의의 이미지에 대해 분류를 수행하는 함수
        :return: None
        '''
        ori_img = Image.open(img_path).convert('L')
        # ori_img.thumbnail((640, 480), Image.ANTIALIAS)
        patches_data = self._create_patch_image(ori_img)
        predict_edges = []

        with tf.Session() as sess:
            self._model = Model(sess)

            sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()
            self._saver.restore(sess, self._FLAGS.trained_param_path)
            logit = self._model.predict(patches_data.reshape(-1, 16, 16, 1) / 255)
            predict_edges = [idx for idx, value in enumerate(logit) if value[0] <= value[1]]
            print(predict_edges)

        predict_img = self._edge_output_on_the_image(ori_img, predict_edges)
        predict_img.show()

    def convert_patch_to_full(self, cam_heatmaps):
        x_pixel, y_pixel = (640, 480)
        x_delta, y_delta, img_cnt = int(x_pixel / 40), int(y_pixel / 30), 0
        newImg = np.empty((x_pixel, y_pixel))

        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                patch_img = cam_heatmaps[img_cnt]
                for y in range(init_y, init_y + y_delta):
                    for x in range(init_x, init_x + x_delta):
                        newImg[x, y] = patch_img[x-init_x][y-init_y]
                img_cnt += 1

        return newImg

    def cam_data_save(self, data):
        with open('D:\\100_dataset\\casia_eyelid_segmentation\\cam_data\\cam_data.txt', mode='a') as f:
            data = data.flatten()
            data = data.astype(np.str_)
            f.write(','.join(data.tolist()) + '\n')

    def cam_predict(self, img_path):
        ori_img = Image.open(img_path).convert('L')
        patches_data = self._create_patch_image(ori_img).reshape(-1, 16, 16, 1) / 255
        predict_edges = []
        cam_heatmaps = []

        with tf.Session(config=self.config) as sess:
            self._model = Model(sess)

            sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()
            self._saver.restore(sess, self._FLAGS.trained_param_path)

            for idx in range(1200):
                print(idx)
                logit, cam_heatmap = self._model.cam(patches_data[idx].reshape(-1, 16, 16, 1))
                predict_edges.append(logit)
                self.cam_data_save(cam_heatmap)
                # cam_heatmaps.append(cam_heatmap)
            predict_edges = [idx for idx, value in enumerate(predict_edges) if value[0] <= value[1]]
            print(predict_edges)

        # predict_img = self._edge_output_on_the_image(ori_img, predict_edges)
        # cam_img = self.convert_patch_to_full(cam_heatmaps)
        #
        # plt.figure(figsize=(10, 8))
        # predict_plot = plt.subplot(121)
        # predict_plot.imshow(predict_img)
        #
        # cam_plot = plt.subplot(122)
        # cam_plot.imshow(ori_img)
        # cam_plot.imshow(cam_img, cmap=plt.cm.jet, alpha=0.5, interpolation='bilinear')
        #
        # plt.show()

if __name__ == '__main__':
    # neuralnet = Neuralnet(is_train=True, save_type='db')
    # neuralnet.train()

    neuralnet = Neuralnet(is_train=False)
    # neuralnet.predict('D:\\111.bmp')
    neuralnet.predict('D:\\100_dataset\\iris\\CASIA\\CASIA-IrisV2\\CASIA-IrisV2\\device1\\0040\\0040_000.bmp')

    # neuralnet = Neuralnet(is_train=False)
    # neuralnet.cam_predict('D:\\100_dataset\\iris\\CASIA\\CASIA-IrisV2\\CASIA-IrisV2\\device1\\0030\\0030_000.bmp')