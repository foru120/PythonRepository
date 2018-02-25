import tensorflow as tf
import numpy as np
import os
import time
from collections import deque
import cx_Oracle
import datetime

from PIL import Image
from PIL import ImageGrab

from Hongbog.SuperResolution.model import Model

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
            stime = time.time()
            self._file_separation()
            etime = time.time()

            if save_type == 'db':
                self._init_database()
                self._get_max_log_num()

    def _flag_setting(self):
        '''
        하이퍼파라미터 값을 설정하는 함수
        :return: None
        '''
        flags = tf.app.flags
        self._FLAGS = flags.FLAGS
        flags.DEFINE_string('blurring_image_path', 'D:\\100_dataset\\casia_blurring\\image_data', '학습 이미지 데이터 경로')
        flags.DEFINE_integer('epochs', 100, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 10, '훈련시 배치 크기')
        flags.DEFINE_integer('max_checks_without_progress', 20, '특정 횟수 만큼 조건이 만족하지 않은 경우')
        flags.DEFINE_string('trained_param_path', 'D:/05_source/PythonRepository/Hongbog/SuperResolution/train_log/0001/image_processing_param.ckpt', '훈련된 파라미터 값 저장 경로')
        flags.DEFINE_string('mon_data_log_path', 'D:/05_source/PythonRepository/Hongbog/SuperResolution//mon_log/mon_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.txt', '훈련시 모니터링 데이터 저장 경로')

    def _init_database(self):
        '''
        데이터베이스 연결을 수행하는 함수
        :return: None
        '''
        self._conn = cx_Oracle.connect('hongbog/hongbog0102@localhost:1521/orcl')

    def _get_cursor(self):
        '''
        데이터베이스 커서를 생성하는 함수
        :return: 데이터베이스 커서, type -> cursor
        '''
        return self._conn.cursor()

    def _close_cursor(self, cur):
        '''
        데이터베이스 커서를 닫는 함수
        :param cur: 닫을 커서
        :return: None
        '''
        cur.close()

    def _close_conn(self):
        '''
        데이터베이스 연결을 해제하는 함수
        :return: None
        '''
        self._conn.close()

    def _file_separation(self):
        file_list = np.array(os.listdir(self._FLAGS.blurring_image_path))
        rand_nums = np.random.permutation(len(file_list))
        cnt = len(file_list)
        self._tot_train_file = file_list[rand_nums[:int(cnt * 0.6)]]
        self._tot_test_file = file_list[rand_nums[int(cnt * 0.6):int(cnt * 0.9)]]
        self._tot_valid_file = file_list[rand_nums[int(cnt * 0.9):]]

    def _data_loading(self, file_name):
        '''
        최종 전처리된 데이터 값을 로딩하는 함수
        :return: None
        '''
        image_size = 64 * 48

        data = np.loadtxt(os.path.join(self._FLAGS.blurring_image_path, file_name), delimiter=',', dtype=np.float32)
        return data[:, :image_size], data[:, image_size:]

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

    def _mon_data_to_file(self, train_acc_mon, train_loss_mon, valid_acc_mon, valid_loss_mon):
        '''
        모니터링 대상(훈련 정확도, 훈련 손실 값, 검증 정확도, 검증 손실 값) 파일로 저장
        :param train_acc_mon: 훈련 정확도
        :param train_loss_mon: 훈련 손실 값
        :param valid_acc_mon: 검증 정확도
        :param valid_loss_mon: 검증 손실 값
        :return: None
        '''
        with open(self._FLAGS.mon_data_log_path, 'a') as f:
            f.write(','.join([str(train_acc_mon), str(valid_acc_mon), str(train_loss_mon), str(valid_loss_mon)]) + '\n')

    def _get_max_log_num(self):
        '''
        현재 로깅된 최대 로그 숫자를 DB 에서 가져오는 함수
        :return: None
        '''
        cur = self._get_cursor()
        cur.execute('select nvl(max(log_num), 0) max_log_num from super_res_log')
        self._max_log_num = cur.fetchone()[0]
        self._close_cursor(cur)

    def _mon_data_to_db(self, train_psnr_mon, train_loss_mon, valid_psnr_mon, valid_loss_mon, train_time):
        '''
        모니터링 대상(훈련 정확도, 훈련 손실 값, 검증 정확도, 검증 손실 값) DB로 저장
        :param train_psnr_mon: 훈련 PSNR
        :param train_loss_mon: 훈련 손실 값
        :param valid_psnr_mon: 검증 PSNR
        :param valid_loss_mon: 검증 손실 값
        :return: None
        '''
        cur = self._get_cursor()
        cur.execute('insert into super_res_log values(:log_num, :log_time, :train_time, :train_psnr, :valid_psnr, :train_loss, :valid_loss)',
                    [self._max_log_num+1, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_time, train_psnr_mon, valid_psnr_mon, train_loss_mon, valid_loss_mon])
        self._conn.commit()
        self._close_cursor(cur)

    def train(self):
        '''
        신경망을 학습하는 함수
        :return: None
        '''
        with tf.Session() as sess:
            self._model = Model(sess)

            sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()

            best_loss_val = np.infty  # 가장 좋은 loss 값을 저장하는 변수
            check_since_last_progress = 0  # early stopping 조건을 만족하지 않은 횟수
            self._best_model_params = None  # 가장 좋은 모델의 parameter 값을 저장하는 변수

            print('Train start!!')

            for epoch in range(self._FLAGS.epochs):
                tot_train_psnr = 0.
                tot_valid_psnr = 0.
                tot_train_loss = 0.
                tot_valid_loss = 0.

                stime = time.time()

                # 훈련 부분
                for train_file in self._tot_train_file:
                    train_x, train_y = self._data_loading(train_file)
                    for idx in range(0, 100, self._FLAGS.batch_size):
                        train_psnr, train_loss, _ = self._model.train(train_x[idx:idx+self._FLAGS.batch_size].reshape(-1, 64, 48, 1), train_y[idx:idx+self._FLAGS.batch_size].reshape(-1, 64, 48, 1))
                        tot_train_psnr += train_psnr / self._FLAGS.batch_size
                        tot_train_loss += train_loss / self._FLAGS.batch_size

                # 검증 부분
                for valid_file in self._tot_valid_file:
                    valid_x, valid_y = self._data_loading(valid_file)
                    for idx in range(0, 100, self._FLAGS.batch_size):
                        valid_psnr, valid_loss = self._model.validation(valid_x[idx:idx+self._FLAGS.batch_size].reshape(-1, 64, 48, 1), valid_y[idx:idx+self._FLAGS.batch_size].reshape(-1, 64, 48, 1))
                        tot_valid_psnr += valid_psnr / self._FLAGS.batch_size
                        tot_valid_loss += valid_loss / self._FLAGS.batch_size

                etime = time.time()

                train_psnr_mon = tot_train_psnr
                train_loss_mon = tot_train_loss
                valid_psnr_mon = tot_valid_psnr
                valid_loss_mon = tot_valid_loss
                train_time = etime - stime

                print('epoch:', epoch + 1, ', train psnr:', round(train_psnr_mon, 2), ', train loss:', round(train_loss_mon, 2),
                      ', validation psnr:', round(valid_psnr_mon, 2), ', validation loss:', round(valid_loss_mon, 2), ', train time:',
                      round(train_time, 2))

                if self.save_type == 'file':
                    self._mon_data_to_file(train_psnr_mon, train_loss_mon, valid_psnr_mon, valid_loss_mon)
                else:
                    self._mon_data_to_db(train_psnr_mon, train_loss_mon, valid_psnr_mon, valid_loss_mon, train_time)

                self.train_acc_mon.append(train_psnr_mon)
                self.train_loss_mon.append(train_loss_mon)
                self.valid_acc_mon.append(valid_psnr_mon)
                self.valid_loss_mon.append(valid_loss_mon)

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

        tot_test_psnr = 0.

        for test_file in self._tot_test_file:
            test_x, test_y = self._data_loading(test_file)
            for idx in range(0, 100, self._FLAGS.batch_size):
                psnr = self._model.get_psnr(test_x[idx:idx+self._FLAGS.batch_size].reshape(-1, 64, 48, 1), test_y[idx:idx+self._FLAGS.batch_size].reshape(-1, 64, 48, 1))
                tot_test_psnr = psnr / self._FLAGS.batch_size

        print('test psnr:', tot_test_psnr)

        self._close_conn()

    def _create_patch_image(self, ori_img):
        '''
        패치 사이즈 단위로 이미지 잘라내는 함수
        :param ori_img: 원본 이미지
        :return: 패치 단위로 분리된 이미지 데이터, type -> ndarray
        '''
        x_pixel, y_pixel = ori_img.size
        x_delta, y_delta = int(x_pixel/10), int(y_pixel/10)
        patches_data = []

        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                img_data = []
                for y in range(init_y, init_y + y_delta):
                    for x in range(init_x, init_x + x_delta):
                        img_data.append(ori_img.getpixel((x, y)))
                patches_data.append(img_data)
        return np.array(patches_data)

    def _merge_super_resolution_image(self, predict_edges):
        '''
        예측된 고해상도 이미지의 패치 들을 가지고 merge 하는 함수
        :param ori_img: 원본 이미지
        :param predict_edges: 예측된 edge 들
        :return: edge 가 표시된 이미지, type -> Image
        '''
        x_pixel, y_pixel = 640, 480
        x_delta, y_delta, pixel_cnt = int(x_pixel / 10), int(y_pixel / 10), 1
        new_img = Image.new('L', (x_pixel, y_pixel))

        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                if img_cnt in predict_edges:
                    for y in range(init_y, init_y + y_delta):
                        for x in range(init_x, init_x + x_delta):
                            new_img.putpixel((x, y), predict_edges[pixel_cnt])
                            pixel_cnt = pixel_cnt + 1

        return new_img

    def predict(self, img_path):
        '''
        임의의 이미지에 대해 분류를 수행하는 함수
        :return: None
        '''
        ori_img = Image.open(img_path)
        patches_data = self._create_patch_image(ori_img)
        predict_pixel = []

        tf.reset_default_graph()

        with tf.Session() as sess:
            self._model = Model(sess)

            sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()
            self._saver.restore(sess, self._FLAGS.trained_param_path)
            logit = self._model.predict(patches_data.reshape(-1, 64, 48, 1))
            predict_pixel = [idx for idx, value in enumerate(logit) if value[0] <= value[1]]

        predict_img = self._merge_super_resolution_image(predict_pixel)
        predict_img.show()

if __name__ == '__main__':
    neuralnet = Neuralnet(is_train=True, save_type='db')
    neuralnet.train()

    # neuralnet = Neuralnet(is_train=False)
    # neuralnet.predict('D:\\Data\\CASIA\\CASIA-IrisV2\\CASIA-IrisV2\\device1\\0008\\0008_001.bmp')