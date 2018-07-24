import tensorflow as tf
import numpy as np
from datetime import datetime
import time

train_file_list = ['tobe_image/image_data_' + str(i) + '.csv' for i in range(1, 5)]
test_file_list = ['tobe_image/image_data_' + str(i) + '.csv' for i in range(5, 7)]

flags = tf.app.flags
FLAGS = flags.FLAGS
FLAGS.image_size = 126
FLAGS.image_color = 1
FLAGS.maxpool_filter_size = 2
FLAGS.num_classes = 2
FLAGS.batch_size = 100
FLAGS.learning_rate = 0.001
FLAGS.epoch = 50

def data_setting(data):
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y = [[1, 0] if y_ == 0 else [0, 1] for y_ in data[:, [-1]]]
    return x, y

def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(data)

    return data_setting(data)

# convolutional network layer 1
def conv1(input_data):  # 126x126 -> 63x63
    # layer 1 (convolutional layer)
    FLAGS.conv1_filter_size = 8
    FLAGS.conv1_layer_size = 3
    FLAGS.stride1 = 1

    with tf.name_scope('conv_1'):
        W_conv1 = tf.Variable(tf.truncated_normal([FLAGS.conv1_filter_size, FLAGS.conv1_filter_size, FLAGS.image_color, FLAGS.conv1_layer_size], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([FLAGS.conv1_layer_size], stddev=0.1))
        h_conv1 = tf.nn.conv2d(input_data, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1_relu = tf.nn.relu(tf.add(h_conv1, b1))
        h_conv1_maxpool = tf.nn.max_pool(h_conv1_relu , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv1_maxpool

# convolutional network layer 2
def conv2(input_data):  # 63x63 -> 32x32
    FLAGS.conv2_filter_size = 5
    FLAGS.conv2_layer_size = 64
    FLAGS.stride2 = 1

    with tf.name_scope('conv_2'):
        W_conv2 = tf.Variable(tf.truncated_normal([FLAGS.conv2_filter_size, FLAGS.conv2_filter_size, FLAGS.conv1_layer_size, FLAGS.conv2_layer_size], stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal([FLAGS.conv2_layer_size], stddev=0.1))
        h_conv2 = tf.nn.conv2d(input_data, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2_relu = tf.nn.relu(tf.add(h_conv2, b2))
        h_conv2_maxpool = tf.nn.max_pool(h_conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv2_maxpool

# cnn 3계층
def conv3(input_data):  # 32x32 -> 16x16
    FLAGS.conv3_filter_size = 3
    FLAGS.conv3_layer_size = 64
    FLAGS.stride3 = 1

    # print('## FLAGS.stride1 ', FLAGS.stride1)
    with tf.name_scope('conv_3'):
        W_conv3 = tf.Variable(tf.truncated_normal([FLAGS.conv3_filter_size, FLAGS.conv3_filter_size, FLAGS.conv2_layer_size, FLAGS.conv3_layer_size], stddev=0.1))
        b3 = tf.Variable(tf.truncated_normal([FLAGS.conv3_layer_size], stddev=0.1))
        h_conv3 = tf.nn.conv2d(input_data, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3_relu = tf.nn.relu(tf.add(h_conv3, b3))
        h_conv3_maxpool = tf.nn.max_pool(h_conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv3_maxpool

# convolutional network layer 3
def conv4(input_data):  # 16x16 -> 8x8
    FLAGS.conv4_filter_size = 5
    FLAGS.conv4_layer_size = 128
    FLAGS.stride4 = 1

    with tf.name_scope('conv_4'):
        W_conv4 = tf.Variable(tf.truncated_normal([FLAGS.conv4_filter_size, FLAGS.conv4_filter_size, FLAGS.conv3_layer_size, FLAGS.conv4_layer_size], stddev=0.1))
        b4 = tf.Variable(tf.truncated_normal([FLAGS.conv4_layer_size], stddev=0.1))
        h_conv4 = tf.nn.conv2d(input_data, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
        h_conv4_relu = tf.nn.relu(tf.add(h_conv4, b4))
        h_conv4_maxpool = tf.nn.max_pool(h_conv4_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv4_maxpool

# fully connected layer 1
def fc1(input_data):
    input_layer_size = 8 * 8 * FLAGS.conv4_layer_size
    FLAGS.fc1_layer_size = 512

    with tf.name_scope('fc_1'):
        # 앞에서 입력받은 다차원 텐서를 fcc에 넣기 위해서 1차원으로 펼치는 작업
        input_data_reshape = tf.reshape(input_data, [-1, input_layer_size])
        W_fc1 = tf.Variable(tf.truncated_normal([input_layer_size, FLAGS.fc1_layer_size], stddev=0.1))
        b_fc1 = tf.Variable(tf.truncated_normal([FLAGS.fc1_layer_size], stddev=0.1))
        h_fc1 = tf.add(tf.matmul(input_data_reshape, W_fc1), b_fc1)  # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc1_relu = tf.nn.relu(h_fc1)

    return h_fc1_relu

# fully connected layer 2
def fc2(input_data):
    FLAGS.fc2_layer_size = 256

    with tf.name_scope('fc_2'):
        W_fc2 = tf.Variable(tf.truncated_normal([FLAGS.fc1_layer_size, FLAGS.fc2_layer_size], stddev=0.1))
        b_fc2 = tf.Variable(tf.truncated_normal([FLAGS.fc2_layer_size], stddev=0.1))
        h_fc2 = tf.add(tf.matmul(input_data, W_fc2), b_fc2)  # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc2_relu = tf.nn.relu(h_fc2)

    return h_fc2_relu

# final layer
# 마지막층
def final_out(input_data):
    with tf.name_scope('final_out'):
        W_fo = tf.Variable(tf.truncated_normal([FLAGS.fc2_layer_size, FLAGS.num_classes], stddev=0.1))
        b_fo = tf.Variable(tf.truncated_normal([FLAGS.num_classes], stddev=0.1))
        h_fo = tf.add(tf.matmul(input_data, W_fo), b_fo)  # h_fc1 = input_data*W_fc1 + b_fc1

    # 최종 레이어에 softmax 함수는 적용하지 않았다.
    return h_fo

# build cnn_graph !
def build_model(images, keep_prob):
    # define CNN network graph
    # output shape will be (*,48,48,16)
    r_cnn1 = conv1(images)  # convolutional layer 1
    # print("shape after cnn1 ", r_cnn1.get_shape())

    # output shape will be (*,24,24,32)
    r_cnn2 = conv2(r_cnn1)  # convolutional layer 2
    # print("shape after cnn2 :", r_cnn2.get_shape())

    # output shape will be (*,12,12,64)
    r_cnn3 = conv3(r_cnn2)  # convolutional layer 3
    # print("shape after cnn3 :", r_cnn3.get_shape())

    # output shape will be (*,6,6,128)
    # 3 or 4 ?
    r_cnn4 = conv4(r_cnn3)  # convolutional layer 4
    # print("shape after cnn4 :", r_cnn4.get_shape())

    # fully connected layer 1
    r_fc1 = fc1(r_cnn4)
    # print("shape after fc1 :", r_fc1.get_shape())

    # fully connected layer2
    r_fc2 = fc2(r_fc1)
    # print("shape after fc2 :", r_fc2.get_shape())

    ## drop out
    # 참고 http://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow
    # 트레이닝시에는 keep_prob < 1.0 , Test 시에는 1.0으로 한다.
    r_dropout = tf.nn.dropout(r_fc2, keep_prob)
    # print("shape after dropout :", r_dropout.get_shape())

    # final layer
    r_out = final_out(r_dropout)
    # print("shape after final layer :", r_out.get_shape())

    return r_out

def main(argv=None):
    images = tf.placeholder(tf.float32, [None, FLAGS.image_size * FLAGS.image_size])
    reshape_images = tf.reshape(images, shape=[-1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
    labels = tf.placeholder(tf.int32, [None, FLAGS.num_classes])

    keep_prob = tf.placeholder(tf.float32)  # dropout 비율
    prediction = build_model(reshape_images, keep_prob)
    # define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

    # define optimizer
    # adam optimizer 이용
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train = optimizer.minimize(loss)

    # for validation
    # with tf.name_scope("prediction"):
    label_max = tf.argmax(labels, 1)
    pre_max = tf.argmax(prediction, 1)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    startTime = datetime.now()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver = tf.train.Saver()  # create saver to store training model into file

        init_op = tf.global_variables_initializer()  # 0.12rc0
        sess.run(init_op)

        ## train
        print('train start -')
        for epoch in range(FLAGS.epoch):
            train_loss = 0.
            sTime = datetime.now()
            for index in range(0, len(train_file_list)):
                total_x, total_y = read_data(train_file_list[index])
                for idx in range(0, 1000, FLAGS.batch_size):
                    batch_X, batch_Y = total_x[idx: idx + FLAGS.batch_size], total_y[idx: idx + FLAGS.batch_size]
                    # keep prob 의경우 대부분의 논문에서 0.7이 적정값이기에 0.7로 수정
                    l, _ = sess.run([loss, train], feed_dict={images: batch_X, labels: batch_Y, keep_prob: 0.7})
                    train_loss += l / FLAGS.batch_size
            eTime = datetime.now()
            print('epoch :', epoch + 1, ', loss :', train_loss, ', time :', eTime - sTime)
        saver.save(sess, 'logs\\' + str(time.time()) + '.ckpt')  # save session

        endTime = datetime.now()
        print('train finish -', endTime - startTime)

        ## test
        test_loss = 0.
        test_accuracy = 0.
        cnt = 0
        for index in range(0, len(test_file_list)):
            total_x, total_y = read_data(test_file_list[index])
            test_len = len(total_y)
            for idx in range(0, 1000, FLAGS.batch_size):
                sample_size = test_len if FLAGS.batch_size > test_len else FLAGS.batch_size
                batch_X, batch_Y = total_x[idx: idx + sample_size], total_y[idx: idx + sample_size]
                l, a = sess.run([loss, accuracy], feed_dict={images: batch_X, labels: batch_Y, keep_prob: 1.0})
                test_loss += l / sample_size
                test_accuracy += a
                cnt += 1
                test_len -= sample_size
        print('test finish - loss :', test_loss, ', accuracy : ', test_accuracy / cnt)

main()