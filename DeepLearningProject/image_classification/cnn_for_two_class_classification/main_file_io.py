from DeepLearningProject.image_classification.classfication_cnn_model.ensemble_cnn_model import Model
import tensorflow as tf
import numpy as np
import time

batch_size = 100

train_file_list = ['data/train_data_' + str(i) + '.csv' for i in range(1, 21)]
test_file_list = ['data/test_data_' + str(i) + '.csv' for i in range(1, 11)]

def data_setting(data):
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y = [[1, 0] if y_ == 0 else [0, 1] for y_ in data[:, [-1]]]
    return x, y

########################################################################################################################
## ▣ Data Loading
##  - 각각의 파일에 대해 load 후 전처리를 수행
########################################################################################################################
def read_data(*filename):
    data1 = np.loadtxt(filename[0], delimiter=',')
    data2 = np.loadtxt(filename[1], delimiter=',')
    data = np.append(data1, data2, axis=0)
    np.random.shuffle(data)

    return data_setting(data)

########################################################################################################################
## ▣ Augmentation Batch - Created by 조원태
##  - 기존 이미지에서 특정 각도(90도)만큼 변경된 데이터 셋을 추가하는 기법
########################################################################################################################
def augment_batch(train_x_batch, train_y_batch):
    rot90_list = []
    batch_size = len(train_y_batch)
    for idx in range(batch_size):
        rot90_list.append(np.rot90(np.asanyarray(train_x_batch[idx]).reshape(126, 126), 1).reshape(1, 126*126))

    agu_list = np.asanyarray(rot90_list).reshape(batch_size, 126*126)
    temp_batch = np.c_[np.append(train_x_batch, agu_list, axis=0), np.append(train_y_batch, train_y_batch, axis=0)]
    np.random.shuffle(temp_batch)
    return temp_batch[:, 0:-2].tolist(), temp_batch[:, -2:].tolist()

########################################################################################################################
## ▣ Data Training
##  - train data : 20,000 개
##  - epoch : 20, batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()

    # agu_data = tl.ImageAugmentation()
    # agu_data.add_random_rotation(max_angle=(-20.0, 20.0))

    models = []
    num_models = 5
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('Learning Started!')

    early_stopping_list = []
    last_epoch = -1
    epoch = 0

    while True:
        avg_cost_list = np.zeros(len(models))
        # train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        for index in range(0, len(train_file_list), 2):
            total_x, total_y = read_data(train_file_list[index], train_file_list[index+1])
            for start_idx in range(0, 2000, batch_size):
                train_x_batch, train_y_batch = total_x[start_idx:start_idx+batch_size], total_y[start_idx:start_idx+batch_size]
                aug_x_batch, aug_y_batch = augment_batch(train_x_batch, train_y_batch)
                for idx, m in enumerate(models):
                    c, _ = m.train(aug_x_batch, aug_y_batch)
                    avg_cost_list[idx] += c / batch_size
                    # train_writer.add_summary(s)

        ################################################################################################################
        ## ▣ early stopping - Created by 배준호
        ##  - prev epoch 과 curr epoch 의 cost 를 비교해서 curr epoch 의 cost 가 더 큰 경우 종료하는 기능
        ################################################################################################################
        saver.save(sess, 'train_log/epoch_' + str(epoch + 1) +'.ckpt')
        early_stopping_list.append(avg_cost_list)
        if len(early_stopping_list) >= 2:
            temp = np.array(early_stopping_list)
            last_epoch = epoch
            if np.sum(temp[0] < temp[1]) > 2:
                print('Epoch: ', '%04d' % (epoch+1), 'cost =', avg_cost_list)
                print('early stopping - epoch({})'.format(epoch+1))
                break
            early_stopping_list.pop(0)
        epoch += 1
        print('Epoch: ', '%04d' % (epoch), 'cost =', avg_cost_list)
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
    num_models = 5
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'train_log/epoch_' + str(epoch) + '.ckpt')

    print('Testing Started!')

    ensemble_accuracy = 0.
    model_accuracy = [0., 0., 0., 0., 0.]
    cnt = 0

    for index in range(0, len(test_file_list), 2):
        total_x, total_y = read_data(test_file_list[index], test_file_list[index+1])
        for start_idx in range(0, 2000, batch_size):
            test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
            test_size = len(test_y_batch)
            predictions = np.zeros(test_size * 10).reshape(test_size, 10)

            model_result = np.zeros(test_size*2, dtype=np.int).reshape(test_size, 2)
            model_result[:, 0] = range(0, test_size)

            for idx, m in enumerate(models):
                model_accuracy[idx] += m.get_accuracy(test_x_batch, test_y_batch)
                p = m.predict(test_x_batch)
                model_result[:, 1] = np.argmax(p, 1)
                for result in model_result:
                    predictions[result[0], result[1]] += 1

            ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
            ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
            cnt += 1
    for i in range(len(model_accuracy)):
        print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
    print('Testing Finished!')