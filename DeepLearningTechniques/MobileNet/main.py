import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageGrab
from drawnow import drawnow

from DeepLearningTechniques.MobileNet.mobilenet import Model
import DeepLearningTechniques.MobileNet.cifar10_pickle as cifar10

epochs = 300
batch_size = 100
value_num = 3

def image_screeshot():
    im = ImageGrab.grab()
    im.show()

# monitoring 관련 parameter
mon_epoch_list = []
mon_value_list = [[] for _ in range(value_num)]
mon_color_list = ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green', 'black']
mon_label_list = ['loss', 'train_acc', 'val_acc']

def monitor_train_cost():
    for cost, color, label in zip(mon_value_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="--", marker="o", label=label)
    plt.title('MobileNet on CIFAR-10')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)

# 모든 변수들의 값을 출력하는 함수
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

# 이전 상태를 restore 하는 함수
def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}  # inputs : 해당 operation 의 입력 데이터를 표현하는 objects
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()

    m = Model(sess, 1e-4)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    best_loss_val = np.infty
    check_since_last_progress = 0  # early stopping 조건을 만족하지 않은 횟수
    max_checks_without_progress = 300  # 특정 횟수 만큼 조건이 만족하지 않은 경우
    best_model_params = None  # 가장 좋은 모델의 parameter 값을 저장하는 변수

    print('Learning Started!')

    '''Data Loading'''
    train_tot_data, train_tot_label = cifar10.load_training_data()
    train_data, train_label = train_tot_data[0: 45000], train_tot_label[0: 45000]
    validation_data, validation_label = train_tot_data[45000: ], train_tot_label[45000: ]
    test_data, test_label = cifar10.load_test_data()

    for epoch in range(epochs):
        epoch_stime = time.time()
        train_accuracy = []
        validation_accuracy = []
        validation_loss = 0.

        '''train part'''
        for start_idx in range(0, len(train_data), batch_size):
            train_x_batch, train_y_batch = train_data[start_idx:start_idx+batch_size], train_label[start_idx:start_idx+batch_size]
            a, _ = m.train(train_x_batch, train_y_batch)
            train_accuracy.append(a)

        '''validation part'''
        for start_idx in range(0, len(validation_data), batch_size):
            validation_x_batch, validation_y_batch = validation_data[start_idx:start_idx + batch_size], validation_label[start_idx:start_idx + batch_size]
            l, a = m.validation(validation_x_batch, validation_y_batch)
            validation_loss += l / batch_size
            validation_accuracy.append(a)

        '''early stopping condition check'''
        if validation_loss < best_loss_val:
            best_loss_val = validation_loss
            check_since_last_progress = 0
            best_model_params = get_model_params()
            saver.save(sess, 'log/mobilenet_cifar10_v1.ckpt')
        else:
            check_since_last_progress += 1

        # monitoring factors
        mon_epoch_list.append(epoch + 1)
        mon_value_list[0].append(validation_loss)
        mon_value_list[1].append(np.mean(np.array(train_accuracy)) * 100)
        mon_value_list[2].append(np.mean(np.array(validation_accuracy)) * 100)

        epoch_etime = time.time()
        print('epoch :', epoch+1, ', loss :', validation_loss, ', train_accuracy :', np.mean(np.array(train_accuracy)),
              ', validation_accuracy :', np.mean(np.array(validation_accuracy)), ', time :', round(epoch_etime-epoch_stime, 6))
        drawnow(monitor_train_cost)

        if check_since_last_progress > max_checks_without_progress:
            print('Early stopping!')
            break

    print('Learning Finished!')

    # 종료 시간 체크
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))

    print('\nTesting Started!')

    if best_model_params:
        restore_model_params(best_model_params)

    test_accuracy = []

    for start_idx in range(0, len(test_data), batch_size):
        test_x_batch, test_y_batch = test_data[start_idx:start_idx + batch_size], test_label[start_idx:start_idx + batch_size]
        a = m.get_accuracy(test_x_batch, test_y_batch)
        test_accuracy.append(a)

    print('Test Accuracy : ', np.mean(np.array(test_accuracy)))
    print('Testing Finished!')

def restore_and_test():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    tf.reset_default_graph()

    with tf.Session(config=config) as sess:
        m = Model(sess, 40)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, 'log/mobilenet_cifar10_v1.ckpt')

        print('Testing Started!')

        test_accuracy = []

        for index in range(0, len(test_file_list)):
            total_x, total_y = read_data(test_file_list[index])
            for start_idx in range(0, 1000, batch_size):
                test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
                a = m.get_accuracy(test_x_batch, test_y_batch)
                test_accuracy.append(a)

        print('Test Accuracy : ', np.mean(np.array(test_accuracy)))
        print('Testing Finished!')