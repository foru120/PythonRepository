from DeepLearningProject.image_classification.classfication_cnn_model.ensemble_cnn_model import Model
import tensorflow as tf
import numpy as np
import time

training_epochs = 20
batch_size = 100
min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size

train_file_list = ['data/train_data_' + str(i) + '.csv' for i in range(1, 21)]
test_file_list = ['data/test_data_' + str(i) + '.csv' for i in range(1, 11)]

def data_setting(data):
    x = (np.array(data[0:-1]) / 255).tolist()
    y = [1, 0] if data[-1:] == 0 else [0, 1]
    return x, y

def read_data(file_list):
    ####################################################################################################################
    ## ▣ Queue Runner
    ##  - 여러개의 데이터 파일을 한번에 메모리에 load 하지 않고, 순차적으로 load 할 수 있게 tensorflow 에서 지원하는 기능.
    ####################################################################################################################
    with tf.device('/cpu:0'):
        filename_queue = tf.train.string_input_producer(file_list, shuffle=False, name='filename_queue')
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        x, y = data_setting(tf.decode_csv(value, record_defaults=list(np.zeros(126*126+1, dtype=np.float32).reshape(126*126+1, 1))))
        train_x_batch, train_y_batch = tf.train.shuffle_batch([x, y], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return train_x_batch, train_y_batch

########################################################################################################################
## ▣ Data Training
##  - train data : 20,000 개
##  - epoch : 20, batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()

    models = []
    num_models = 5
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())

    print('Learning Started!')

    train_x_batch, train_y_batch = read_data(train_file_list)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in range(training_epochs):
        avg_cost_list = np.zeros(len(models))
        # train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        total_batch_cnt = int(20000 / batch_size)
        for i in range(0, total_batch_cnt):
            train_xs, train_ys = sess.run([train_x_batch, train_y_batch])
            # 각각의 모델 훈련
            for idx, m in enumerate(models):
                c, _ = m.train(train_xs, train_ys)
                avg_cost_list[idx] += c / batch_size
                # train_writer.add_summary(s)
        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

    coord.request_stop()
    coord.join(threads)
    print('Learning Finished!')

    # 종료 시간 체크
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))

########################################################################################################################
## ▣ Data Test
##  - test data : 10,000 개
##  - epoch : 20, batch_size : 100, model : 5개
########################################################################################################################
    print('Testing Started!')

    test_x_batch, test_y_batch = read_data(test_file_list)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    total_batch_cnt = int(10000 / batch_size)
    ensemble_accuracy = 0.
    for i in range(0, total_batch_cnt):
        test_xs, test_ys = sess.run([test_x_batch, test_y_batch])
        test_size = len(test_ys)
        predictions = np.zeros(test_size * 10).reshape(test_size, 10)

        model_result = np.zeros(test_size*2, dtype=np.int).reshape(test_size, 2)
        model_result[:, 0] = range(0, test_size)

        for idx, m in enumerate(models):
            # print(idx, 'Accuracy: ', m.get_accuracy(test_xs, test_ys))
            p = m.predict(test_xs)
            model_result[:, 1] = np.argmax(p, 1)
            for result in model_result:
                predictions[result[0], result[1]] += 1

        ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_ys, 1))
        ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / total_batch_cnt)

    coord.request_stop()
    coord.join(threads)

    print('Testing Finished!')