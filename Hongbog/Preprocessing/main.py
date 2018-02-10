import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageGrab

from Hongbog.Preprocessing.model import Model

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

    def __init__(self, is_train):
        self._flag_setting()  # flag setting

        if is_train == True:
            tot_data = []
            for idx, file_name in enumerate(os.listdir(self._FLAGS.image_data_path)):
                data = np.loadtxt(os.path.join(self._FLAGS.image_data_path, file_name), delimiter=',')
                if idx == 0:
                    tot_data = data
                else:
                    tot_data = np.concatenate((tot_data, data), axis=0)
            # np.random.shuffle(tot_data)
            self._tot_x, self._tot_y = tot_data[:, 0:-1], tot_data[:, -1]
            self._data_loading()  # data loading

    def _flag_setting(self):
        '''
        하이퍼파라미터 값을 설정하는 함수
        :return: None
        '''
        flags = tf.app.flags
        self._FLAGS = flags.FLAGS
        flags.DEFINE_string('image_data_path', 'D:\\Data\\casia_preprocessing\\image_data', '훈련 이미지 데이터 경로')
        flags.DEFINE_integer('epochs', 100, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 100, '훈련시 배치 크기')
        flags.DEFINE_integer('max_checks_without_progress', 20, '특정 횟수 만큼 조건이 만족하지 않은 경우')
        flags.DEFINE_string('trained_param_path', 'log/0003/image_processing_param.ckpt', '훈련된 파라미터 값이 저장된 경로')

    def _data_loading(self):
        '''
        로딩된 전체 데이터에 대해 훈련 데이터, 테스트 데이터, 검증 데이터로 분리하는 함수
        :return: None
        '''
        train_end_idx, test_end_idx, valid_end_idx = int(self._tot_x.shape[0] * 0.6), int(self._tot_x.shape[0] * 0.9), int(self._tot_x.shape[0])
        self._train_x, self._train_y = self._tot_x[0:train_end_idx, ], self._tot_y[0:train_end_idx, ]
        self._test_x,  self._test_y  = self._tot_x[train_end_idx:test_end_idx, ], self._tot_y[train_end_idx:test_end_idx, ]
        self._valid_x, self._valid_y = self._tot_x[test_end_idx:, ], self._tot_y[test_end_idx:, ]

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
        with tf.Session() as sess:
            self._model = Model(sess, 40)

            sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()

            best_loss_val = np.infty  # 가장 좋은 loss 값을 저장하는 변수
            check_since_last_progress = 0  # early stopping 조건을 만족하지 않은 횟수
            self._best_model_params = None  # 가장 좋은 모델의 parameter 값을 저장하는 변수

            for epoch in range(self._FLAGS.epochs):
                train_acc_list = []
                valid_acc_list = []
                tot_train_loss = 0.
                tot_valid_loss = 0.

                stime = time.time()

                # 훈련 부분
                for idx in range(0, self._train_x.shape[0], self._FLAGS.batch_size):
                    train_x_batch, train_y_batch = self._train_x[idx:idx+self._FLAGS.batch_size, ], self._train_y[idx:idx+self._FLAGS.batch_size, ]
                    if epoch+1 in (50, 75):  # dynamic learning rate
                        self._model.learning_rate = self._model.learning_rate/10
                    train_acc, train_loss, _ = self._model.train(train_x_batch.reshape(-1, 16, 16, 1), train_y_batch)
                    tot_train_loss += train_loss / self._FLAGS.batch_size
                    train_acc_list.append(train_acc)

                # 검증 부분
                for idx in range(0, self._valid_x.shape[0], self._FLAGS.batch_size):
                    valid_x_batch, valid_y_batch = self._valid_x[idx:idx+self._FLAGS.batch_size, ], self._valid_y[idx:idx+self._FLAGS.batch_size, ]
                    valid_loss, valid_acc = self._model.validation(valid_x_batch.reshape(-1, 16, 16, 1), valid_y_batch)
                    tot_valid_loss += valid_loss / self._FLAGS.batch_size
                    valid_acc_list.append(valid_acc)

                etime = time.time()

                print('epoch:', epoch+1, ', train accuracy:', np.mean(np.array(train_acc_list)), ', train loss:', tot_train_loss, ', validation accuracy:', np.mean(np.array(valid_acc_list)), ', validation loss:', tot_valid_loss, ', train time:', etime-stime)

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
        신경망을 테스트하는 함수
        :return: None
        '''
        if self._best_model_params:  # 가장 좋은 신경망의 파라미터 값을 Restore
            self._restore_model_params()

        test_acc_list = []

        for idx in range(0, self._test_x.shape[0], self._FLAGS.batch_size):
            test_x_batch, test_y_batch = self._test_x[idx:idx+self._FLAGS.batch_size, ], self._test_y[idx:idx+self._FLAGS.batch_size, ]
            a = self._model.get_accuracy(test_x_batch.reshape(-1, 16, 16, 1), test_y_batch)
            test_acc_list.append(a)

        print('test accuracy:', np.mean(np.array(test_acc_list)))

    def create_patch_image(self, img_path):
        '''
        패치 사이즈 단위로 이미지 잘라내는 함수
        :param img_path: 패치 단위로 분리할 이미지 경로
        :return: 패치 단위로 분리된 이미지 데이터, type -> ndarray
        '''
        img = Image.open(img_path)
        x_pixel, y_pixel = img.size
        x_delta, y_delta = int(x_pixel/40), int(y_pixel/30)
        patches_data = []

        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                img_data = []
                for y in range(init_y, init_y + y_delta):
                    for x in range(init_x, init_x + x_delta):
                        img_data.append(img.getpixel((x, y)))
                patches_data.append(img_data)
        return np.array(patches_data)

    def predict(self, img_path):
        '''
        임의의 이미지에 대해 분류를 수행하는 함수
        :return: None
        '''
        patches_data = self.create_patch_image(img_path)

        tf.reset_default_graph()

        with tf.Session() as sess:
            self._model = Model(sess, 40)

            sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()
            self._saver.restore(sess, self._FLAGS.trained_param_path)
            logit = self._model.predict(patches_data.reshape(-1, 16, 16, 1))
            print([idx for idx, value in enumerate(logit) if value[0] <= value[1]])

neuralnet = Neuralnet(is_train=True)
neuralnet.train()

# neuralnet = Neuralnet(is_train=False)
# neuralnet.predict('D:\\Data\\CASIA\\CASIA-IrisV2\\CASIA-IrisV2\\device1\\0029\\0029_000.bmp')