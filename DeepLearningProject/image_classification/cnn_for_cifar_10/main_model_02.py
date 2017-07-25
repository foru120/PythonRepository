from DeepLearningProject.image_classification.cnn_for_cifar_10.ensemble_cnn_model_cifar_10 import Model
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from drawnow import drawnow
from PIL import ImageGrab

batch_size = 100
num_models = 5

train_file_list = ['data/train_data_' + str(i) + '.csv' for i in range(1, 51)]
test_file_list = ['data/test_data_' + str(i) + '.csv' for i in range(1, 11)]

def data_setting(data):
    # x : 데이터, y : 라벨
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y_tmp = np.zeros([len(data), 10])
    for i in range(0, len(data)):
        label = int(data[i][-1])
        y_tmp[i, label - 1] = 1
    y = y_tmp.tolist()

    return x, y

def read_data(filename):
    ####################################################################################################################
    ## ▣ Data Loading
    ##  - 각각의 파일에 대해 load 후 전처리를 수행
    ####################################################################################################################
    data = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(data)
    return data_setting(data)

def image_screeshot():
    im = ImageGrab.grab()
    im.show()

# monitoring 관련 parameter
mon_epoch_list = []
mon_cost_list = [[] for m in range(num_models)]
mon_color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
mon_label_list = ['model'+str(m+1) for m in range(num_models)]

################################################################################################################
## ▣ Train Monitoring - Created by 박상범
##  - 실시간으로 train cost 값을 monitoring 하는 기능
################################################################################################################
def monitor_train_cost():
    for cost, color, label in zip(mon_cost_list, mon_color_list[0:len(mon_label_list)], mon_label_list):
        plt.plot(mon_epoch_list, cost, c=color, lw=2, ls="--", marker="o", label=label)
    plt.title('Epoch per Cost Graph')
    plt.legend(loc=1)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)

########################################################################################################################
## ▣ Data Training
##  - train data : 50,000 개 (10클래스, 클래스별 5,000개)
##  - epoch : 20, batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()
    models = []
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('Learning Started!')

    early_stopping_list = []
    last_epoch = -1
    epoch = 0
    early_stop_count = 0

    while True:
        avg_cost_list = np.zeros(len(models))
        for index in range(0, len(train_file_list)):
            total_x, total_y = read_data(train_file_list[index])
            for start_idx in range(0, 500, batch_size):
                train_x_batch, train_y_batch = total_x[start_idx:start_idx+batch_size], total_y[start_idx:start_idx+batch_size]

                for idx, m in enumerate(models):
                    c, _ = m.train(train_x_batch, train_y_batch)
                    avg_cost_list[idx] += c / batch_size

        mon_epoch_list.append(epoch + 1)
        for idx, cost in enumerate(avg_cost_list):
            mon_cost_list[idx].append(cost)
        drawnow(monitor_train_cost)

        ################################################################################################################
        ## ▣ early stopping - Created by 배준호
        ##  - prev epoch 과 curr epoch 의 cost 를 비교해서 curr epoch 의 cost 가 더 큰 경우 종료하는 기능
        ################################################################################################################
        saver.save(sess, 'log/epoch_' + str(epoch + 1) +'.ckpt')
        early_stopping_list.append(avg_cost_list)
        diff = 0
        if len(early_stopping_list) >= 2:
            temp = np.array(early_stopping_list)
            last_epoch = epoch
            diff = np.sum(temp[0] < temp[1])
            if diff > 2:
                print('Epoch: ', '%04d' % (epoch+1), 'cost =', avg_cost_list)
                print('early stopping - epoch({})'.format(epoch+1), ' - ', diff)
                early_stop_count += 1
                if early_stop_count == 3:
                    break
            early_stopping_list.pop(0)
        epoch += 1
        print('Epoch: ', '%04d' % (epoch), 'cost =', avg_cost_list, ' - ', diff)

    print('Learning Finished!')

    # 종료 시간 체크
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))

tf.reset_default_graph()

########################################################################################################################
## ▣ Data Test
##  - test data : 10,000 개
##  - batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    models = []
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'log/epoch_' + str(last_epoch) + '.ckpt')

    print('Testing Started!')

    ensemble_accuracy = 0.
    ensemble_accuracy_old = 0.
    model_accuracy = [0., 0., 0., 0., 0.]
    cnt = 0

    for index in range(0, len(test_file_list)):
        total_x, total_y = read_data(test_file_list[index])
        for start_idx in range(0, 500, batch_size):
            test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
            test_size = len(test_y_batch)
            predictions = np.zeros((test_size, 10, 3))
            predictions[:, :, 0] = range(0, 10)

            model_result = np.zeros(test_size, dtype='int32')

            for idx, m in enumerate(models):
                model_accuracy[idx] += m.get_accuracy(test_x_batch, test_y_batch)
                p = m.predict(test_x_batch)
                model_result[:] = np.argmax(p, 1)
                for idx, result in enumerate(model_result):
                    predictions[idx, result, 1] += 1
                predictions[:, :, 2] += p

            predictions_old = np.array(predictions[:, :, 1])  # for get vote-only Accuracy

            # sort
            predictions.view('i8, i8, i8').sort(order=['f1', 'f2'], axis=1)
            res = np.array(predictions[:, -1, 0], dtype='int64')

            # get Accuracy
            ensemble_correct_prediction = tf.equal(res, tf.argmax(test_y_batch, 1))
            ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

            # get vote-only Accuracy
            ensemble_correct_prediction_old = tf.equal(tf.argmax(predictions_old, 1), tf.argmax(test_y_batch, 1))
            ensemble_accuracy_old += tf.reduce_mean(tf.cast(ensemble_correct_prediction_old, tf.float32))
            cnt += 1
    for i in range(len(model_accuracy)):
        print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
    print('Testing Finished!')