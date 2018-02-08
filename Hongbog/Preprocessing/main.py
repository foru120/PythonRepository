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
    IMAGE_DATA_PATH = 'D:\\Data\\casia_preprocessing\\image_data'
    EPOCHS = 50
    BATCH_SIZE = 100
    tot_x, tot_y = None, None
    train_x, train_y = None, None
    test_x, test_y = None, None
    valid_x, valid_y = None, None

    def __init__(self):
        tot_data = []
        for idx, file_name in enumerate(os.listdir(IMAGE_DATA_PATH)):
            data = np.loadtxt(os.path.join(IMAGE_DATA_PATH, file_name), delimiter=',')
            if idx == 0:
                tot_data = data
            else:
                tot_data = np.concatenate((self.tot_data, data), axis=0)
        np.random.shuffle(self.tot_data)
        tot_x, tot_y = tot_data[:, 0:-1], tot_data[:, -1]
        self.data_loading()

    def data_loading(self):
        train_end_idx, test_end_idx, valid_end_idx = int(x.shape[0] * 0.6), int(x.shape[0] * 0.9), int(x.shape[0])
        self.train_x, self.train_y = x[0:train_end_idx, ], y[0:train_end_idx, ]
        self.test_x, self.test_y = x[train_end_idx:test_end_idx, ], y[train_end_idx:test_end_idx, ]
        self.valid_x, self.valid_y = x[test_end_idx:, ], y[test_end_idx:, ]

    def train(self):
        pass
# def image_screeshot():
#     im = ImageGrab.grab()
#     im.show()
#
# # monitoring 관련 parameter
# mon_epoch_list = []
# mon_value_list = [[] for _ in range(value_num)]
# mon_color_list = ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green', 'black']
# mon_label_list = ['loss', 'train_acc', 'val_acc']
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
# # 모든 변수들의 값을 출력하는 함수
# def get_model_params():
#     gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#     return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}
#
# # 이전 상태를 restore 하는 함수
# def restore_model_params(model_params):
#     gvar_names = list(model_params.keys())
#     assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
#     init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}  # inputs : 해당 operation 의 입력 데이터를 표현하는 objects
#     feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
#     tf.get_default_session().run(assign_ops, feed_dict=feed_dict)
#
# ########################################################################################################################
# ## ▣ Data Training
# ##  - train data : 50,000 개 (10클래스, 클래스별 5,000개)
# ##  - epoch : 100, batch_size : 100, model : 1개
# ########################################################################################################################
# # config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.8
#
# with tf.Session() as sess:
#     # 시작 시간 체크
#     stime = time.time()
#
#     m = Model(sess, 40)
#
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#
#     best_loss_val = np.infty
#     check_since_last_progress = 0  # early stopping 조건을 만족하지 않은 횟수
#     max_checks_without_progress = 100  # 특정 횟수 만큼 조건이 만족하지 않은 경우
#     best_model_params = None  # 가장 좋은 모델의 parameter 값을 저장하는 변수
#
#     print('Learning Started!')
#
#     for epoch in range(epochs):
#         epoch_stime = time.time()
#         train_accuracy = []
#         validation_accuracy = []
#         validation_loss = 0.
#
#         '''train part'''
#         for index in range(0, len(train_file_list)):
#             total_x, total_y = read_data(train_file_list[index])
#             for start_idx in range(0, 1000, batch_size):
#                 train_x_batch, train_y_batch = total_x[start_idx:start_idx+batch_size], total_y[start_idx:start_idx+batch_size]
#                 if epoch+1 in (50, 75):  # dynamic learning rate
#                     m.learning_rate = m.learning_rate/10
#                 a, _ = m.train(train_x_batch, train_y_batch)
#                 train_accuracy.append(a)
#
#         '''validation part'''
#         for index in range(0, len(validation_file_list)):
#             total_x, total_y = read_data(validation_file_list[index])
#             for start_idx in range(0, 1000, batch_size):
#                 validation_x_batch, validation_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
#                 l, a = m.validation(validation_x_batch, validation_y_batch)
#                 validation_loss += l / batch_size
#                 validation_accuracy.append(a)
#
#         '''early stopping condition check'''
#         if validation_loss < best_loss_val:
#             best_loss_val = validation_loss
#             check_since_last_progress = 0
#             best_model_params = get_model_params()
#             saver.save(sess, 'log/densenet_cifar10_v2.ckpt')
#         else:
#             check_since_last_progress += 1
#
#         # monitoring factors
#         mon_epoch_list.append(epoch + 1)
#         mon_value_list[0].append(validation_loss)
#         mon_value_list[1].append(np.mean(np.array(train_accuracy)) * 100)
#         mon_value_list[2].append(np.mean(np.array(validation_accuracy)) * 100)
#
#         epoch_etime = time.time()
#         print('epoch :', epoch+1, ', loss :', validation_loss, ', train_accuracy :', np.mean(np.array(train_accuracy)),
#               ', validation_accuracy :', np.mean(np.array(validation_accuracy)), ', time :', round(epoch_etime-epoch_stime, 6))
#         drawnow(monitor_train_cost)
#
#         if check_since_last_progress > max_checks_without_progress:
#             print('Early stopping!')
#             break
#
#     print('Learning Finished!')
#
#     # 종료 시간 체크
#     etime = time.time()
#     print('consumption time : ', round(etime-stime, 6))
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