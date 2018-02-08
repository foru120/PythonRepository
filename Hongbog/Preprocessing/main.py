import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import re
from PIL import ImageGrab
import os
import time

from Hongbog.Preprocessing.model import Model

class Training:
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

    def __init__(self):
        self._flag_setting()  # flag setting
        tot_data = []
        for idx, file_name in enumerate(os.listdir(self._FLAGS.image_data_path)):
            data = np.loadtxt(os.path.join(self._FLAGS.image_data_path, file_name), delimiter=',')
            if idx == 0:
                tot_data = data
            else:
                tot_data = np.concatenate((tot_data, data), axis=0)
        np.random.shuffle(tot_data)
        self._tot_x, self._tot_y = tot_data[:, 0:-1], tot_data[:, -1]
        self._data_loading()  # data loading

    def _flag_setting(self):
        '''
        훈련시 사용할 하이퍼파라미터를 설정하는 함수
        :return: None
        '''
        flags = tf.app.flags
        self._FLAGS = flags.FLAGS
        flags.DEFINE_string('image_data_path', 'D:\\Data\\casia_preprocessing\\image_data', '이미지 데이터 경로')
        flags.DEFINE_integer('epochs', 100, '훈련시 에폭 수')
        flags.DEFINE_integer('batch_size', 100, '훈련시 배치 크기')
        flags.DEFINE_integer('max_checks_without_progress', 100, '특정 횟수 만큼 조건이 만족하지 않은 경우')

    def _data_loading(self):
        '''
        로딩된 전체 데이터에 대해 훈련 데이터, 테스트 데이터, 검증 데이터로 분리하는 함수
        :return: None
        '''
        train_end_idx, test_end_idx, valid_end_idx = int(self._tot_x.shape[0] * 0.6), int(self._tot_x.shape[0] * 0.9), int(self._tot_x.shape[0])
        self._train_x, self._train_y = self._tot_x[0:train_end_idx, ], self._tot_y[0:train_end_idx, ]
        self._test_x,  self._test_y  = self._tot_x[train_end_idx:test_end_idx, ], self._tot_y[train_end_idx:test_end_idx, ]
        self._valid_x, self._valid_y = self._tot_x[test_end_idx:, ], self._tot_y[test_end_idx:, ]

    def _get_model_params():
        '''
        텐서플로우 내에 존재하는 모든 파라미터들의 값을 추출하는 함수
        :return: 텐서플로우 내의 {변수: 값}, type -> Dict
        '''
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

    def _restore_model_params(model_params):
        '''
        메모리에 존재하는 파라미터들의 값을 Restore 하는 함수
        :return: None
        '''
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}  # inputs : 해당 operation 의 입력 데이터를 표현하는 objects
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

    def train(self):
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
                tot_valid_loss = 0.

                # 훈련 부분
                for idx in range(0, self._train_x.shape[0], self._FLAGS.batch_size):
                    train_x_batch, train_y_batch = self._train_x[idx:idx+self._FLAGS.batch_size, ], self._train_y[idx:idx+self._FLAGS.batch_size, ]
                    if epoch+1 in (50, 75):  # dynamic learning rate
                        self._model.learning_rate = self._model.learning_rate/10
                        train_acc, _ = self._model.train(train_x_batch, train_y_batch)
                        train_acc_list.append(train_acc)

                # 검증 부분
                for idx in range(0, self._valid_x.shape[0], self._FLAGS.batch_size):
                    valid_x_batch, valid_y_batch = self._valid_x[idx:idx+self._FLAGS.batch_size, ], self._valid_y[idx:idx+self._FLAGS.batch_size, ]
                    valid_loss, valid_acc = self._model.validation(valid_x_batch, valid_y_batch)
                    tot_valid_loss += valid_loss / self._FLAGS.batch_size
                    valid_acc_list.append(valid_acc)

                # Early Stopping 조건 확인
                if tot_valid_loss < best_loss_val:
                    best_loss_val = tot_valid_loss
                    check_since_last_progress = 0
                    self._best_model_params = self._get_model_params()
                    self._saver.save(sess, 'log/image_processing_param.ckpt')
                else:
                    check_since_last_progress += 1

                if check_since_last_progress > self._FLAGS.max_checks_without_progress:
                    print('Early stopping!')
                    break

            print('Learning Finished!')

# def image_screeshot():
#     im = ImageGrab.grab()
#     im.show()
#
# def monitor_train_cost():
#     for cost, color, label in zip(mon_value_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
#         plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="--", marker="o", label=label)
#     plt.title('DenseNet-BC on CIFAR-10')
#     plt.legend(loc=1)
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
#     plt.grid(True)
#
#     print('\nTesting Started!')
#
#     if best_model_params:
#         restore_model_params(best_model_params)
#
#     test_accuracy = []
#
#     for index in range(0, len(test_file_list)):
#         total_x, total_y = read_data(test_file_list[index])
#         for start_idx in range(0, 1000, batch_size):
#             test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
#             a = m.get_accuracy(test_x_batch, test_y_batch)
#             test_accuracy.append(a)
#
#     print('Test Accuracy : ', np.mean(np.array(test_accuracy)))
#     print('Testing Finished!')
#
# def restore_and_test():
#     tf.reset_default_graph()
#
#     with tf.Session(config=config) as sess:
#         m = Model(sess, 40)
#
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         saver.restore(sess, 'log/densenet_cifar10_v3.ckpt')
#
#         print('Testing Started!')
#
#         test_accuracy = []
#
#         for index in range(0, len(test_file_list)):
#             total_x, total_y = read_data(test_file_list[index])
#             for start_idx in range(0, 1000, batch_size):
#                 test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
#                 a = m.get_accuracy(test_x_batch, test_y_batch)
#                 test_accuracy.append(a)
#
#         print('Test Accuracy : ', np.mean(np.array(test_accuracy)))
#         print('Testing Finished!')