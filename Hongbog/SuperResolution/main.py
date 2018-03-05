import tensorflow as tf
import numpy as np
import time
from collections import deque

from PIL import Image
from PIL import ImageGrab

from Hongbog.SuperResolution.model import Model
from Hongbog.SuperResolution.data_loader import Dataloader
from Hongbog.SuperResolution.database import Database

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

        if (is_train == True) and (save_type == 'db'):  # data loading
            self.db = Database()
            self.db.init_database()
            self.db.get_max_log_num()

    def _flag_setting(self):
        '''
        하이퍼파라미터 값을 설정하는 함수
        :return: None
        '''
        flags = tf.app.flags
        self._FLAGS = flags.FLAGS
        flags.DEFINE_string('blurring_image_path', 'D:\\Data\\casia_blurring\\image_data', '학습 이미지 데이터 경로')
        flags.DEFINE_integer('epochs', 100, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 10, '훈련시 배치 크기')
        flags.DEFINE_integer('max_checks_without_progress', 20, '특정 횟수 만큼 조건이 만족하지 않은 경우')
        flags.DEFINE_string('trained_param_path', 'D:/Source/PythonRepository/Hongbog/SuperResolution/train_log/0002/image_processing_param.ckpt', '훈련된 파라미터 값 저장 경로')
        flags.DEFINE_integer('data_cnt_per_file', 100, '파일당 데이터 개수')

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
        '''
        신경망을 학습하는 함수
        :return: None
        '''
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1)
        )

        with tf.Session(config=config) as sess:
            self._model = Model(sess)

            sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()

            ck_point = tf.train.get_checkpoint_state(self._FLAGS.trained_param_path)

            if ck_point:
                self._saver.restore(sess, self._FLAGS.trained_param_path)

            self.loader = Dataloader(data_path='D:\\Data\\casia_blurring\\image_data', batch_size=self._FLAGS.batch_size)
            self.loader.start()

            best_loss_val = np.infty  # 가장 좋은 loss 값을 저장하는 변수
            check_since_last_progress = 0  # early stopping 조건을 만족하지 않은 횟수
            self._best_model_params = None  # 가장 좋은 모델의 parameter 값을 저장하는 변수

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print('Train start!!')

            for epoch in range(self._FLAGS.epochs):
                tot_train_psnr_list = []
                tot_valid_psnr_list = []
                tot_train_loss = 0.
                tot_valid_loss = 0.

                stime = time.time()

                # 훈련 부분
                for idx in range(self.loader.train_cnt * int(self._FLAGS.data_cnt_per_file / self._FLAGS.batch_size)):
                    s_batch_time = time.time()
                    train_x, train_y = sess.run([self.loader.train_x, self.loader.train_y])
                    train_psnr, train_loss, _ = self._model.train(train_x.reshape(-1, 64, 48, 1), train_y.reshape(-1, 64, 48, 1))
                    tot_train_psnr_list.append(train_psnr)
                    tot_train_loss += train_loss / self._FLAGS.batch_size
                    e_batch_time = time.time()
                    print('[Training] [' + str(int(idx % self._FLAGS.batch_size) + 1) + '/10] Epoch: ' + str(epoch) + ', PSNL: ' + str(train_psnr) + ', Loss: ' + str(train_loss) + ', Time(s): ' + str(round(e_batch_time - s_batch_time, 2)))

                # 검증 부분
                for idx in range(self.loader.valid_cnt * int(self._FLAGS.data_cnt_per_file / self._FLAGS.batch_size)):
                    s_batch_time = time.time()
                    valid_x, valid_y = sess.run([self.loader.valid_x, self.loader.valid_y])
                    valid_psnr, valid_loss = self._model.validation(valid_x.reshape(-1, 64, 48, 1), valid_y.reshape(-1, 64, 48, 1))
                    tot_valid_psnr_list.append(valid_psnr)
                    tot_valid_loss += valid_loss / self._FLAGS.batch_size
                    e_batch_time = time.time()
                    print('[Validation] [' + str(int(idx % self._FLAGS.batch_size) + 1) + '/10] Epoch: ' + str(epoch) + ', PSNL: ' + str(train_psnr) + ', Loss: ' + str(train_loss) + ', Time(s): ' + str(round(e_batch_time - s_batch_time, 2)))

                etime = time.time()

                train_psnr_mon = np.mean(np.array(tot_train_psnr_list), dtype=np.float64)
                train_loss_mon = tot_train_loss
                valid_psnr_mon = np.mean(np.array(tot_valid_psnr_list), dtype=np.float64)
                valid_loss_mon = tot_valid_loss
                train_time = etime - stime

                print('epoch:', epoch + 1, ', train psnr:', round(train_psnr_mon, 2), ', train loss:', round(train_loss_mon, 2), ', validation psnr:', round(valid_psnr_mon, 2), ', validation loss:', round(valid_loss_mon, 2), ', train time:', round(train_time, 2))

                if self.save_type == 'file':
                    self.db.mon_data_to_file(train_psnr_mon, train_loss_mon, valid_psnr_mon, valid_loss_mon)
                else:
                    self.db.mon_data_to_db(train_psnr_mon, train_loss_mon, valid_psnr_mon, valid_loss_mon, train_time)

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

            self._test(sess)

            coord.request_stop()
            coord.join(threads)

    def _test(self, sess):
        '''
        테스트 데이터에 대해 분류를 수행하는 함수
        :return: None
        '''
        print('Test start!!!')
        if self._best_model_params:  # 가장 좋은 신경망의 파라미터 값을 Restore
            self._restore_model_params()

        tot_test_psnr_list = []

        for _ in range(self.loader.test_cnt * int(self._FLAGS.data_cnt_per_file / self._FLAGS.batch_size)):
            test_x, test_y = sess.run([self.loader.test_x, self.loader.test_y])
            test_psnr = self._model.get_psnr(test_x.reshape(-1, 64, 48, 1), test_y.reshape(-1, 64, 48, 1))
            tot_test_psnr_list.append(test_psnr)

        print('test psnr:', np.mean(np.array(tot_test_psnr_list)))

        self.db.close_conn()

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

    def _merge_super_resolution_image(self):
        '''
        예측된 고해상도 이미지의 패치 들을 가지고 merge 하는 함수
        :param ori_img: 원본 이미지
        :param predict_edges: 예측된 edge 들
        :return: edge 가 표시된 이미지, type -> Image
        '''
        x_pixel, y_pixel = 640, 480
        x_delta, y_delta, pixel_cnt, img_cnt = int(x_pixel / 10), int(y_pixel / 10), 0, 0
        new_img = Image.new('L', (x_pixel, y_pixel))

        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                pixel_cnt = 0
                value = self.logit[img_cnt].ravel()
                for y in range(init_y, init_y + y_delta):
                    for x in range(init_x, init_x + x_delta):
                        new_img.putpixel((x, y), int(value[pixel_cnt]))
                        pixel_cnt = pixel_cnt + 1
                img_cnt = img_cnt + 1
        return new_img

    def predict(self, img_path):
        '''
        임의의 이미지에 대해 고해상도 이미지로 변환하는 함수
        :return: None
        '''
        ori_img = Image.open(img_path)
        patches_data = self._create_patch_image(ori_img)

        tf.reset_default_graph()

        with tf.Session() as sess:
            self._model = Model(sess)

            sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()
            self._saver.restore(sess, self._FLAGS.trained_param_path)
            self.logit = self._model.predict(patches_data.reshape(-1, 64, 48, 1))

        predict_img = self._merge_super_resolution_image()
        predict_img.show()

if __name__ == '__main__':
    # neuralnet = Neuralnet(is_train=True, save_type='db')
    # neuralnet.train()

    neuralnet = Neuralnet(is_train=False)
    neuralnet.predict('D:\\Data\\casia_original\\non-cropped\\device1\\0000\\0000_000.bmp')