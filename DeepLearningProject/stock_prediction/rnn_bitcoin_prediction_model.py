import os
import numpy as np
import tensorflow as tf
import time
import re
import matplotlib.pyplot as plt

class RNN_Model:
    def __init__(self, sess, n_inputs, n_sequences, n_hiddens, n_outputs, hidden_layer_cnt, file_name, model_name):
        self.sess = sess
        self.n_inputs = n_inputs
        self.n_sequences = n_sequences
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.hidden_layer_cnt = hidden_layer_cnt
        self.file_name = file_name
        self.model_name = model_name
        self.regularizer = tf.contrib.layers.l2_regularizer(0.001)
        self.training = True
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.model_name):
            self.learning_rate = 0.001

            self.X = tf.placeholder(tf.float32, [None, self.n_sequences, self.n_inputs])
            self.Y = tf.placeholder(tf.float32, [None, self.n_outputs])

            self.multi_cells = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(self.n_hiddens) for _ in range(self.hidden_layer_cnt)], state_is_tuple=True)
            self.outputs, _states = tf.nn.dynamic_rnn(self.multi_cells, self.X, dtype=tf.float32)
            self.Y_ = tf.contrib.layers.fully_connected(self.outputs[:, -1], self.n_outputs, activation_fn=None)
            self.reg_loss = tf.reduce_sum([self.regularizer(train_var) for train_var in tf.trainable_variables() if re.search('(kernel)|(weights)', train_var.name) is not None])
            self.loss = tf.reduce_sum(tf.square(self.Y_ - self.Y)) + self.reg_loss
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.targets = tf.placeholder(tf.float32, [None, 1])
            self.predictions = tf.placeholder(tf.float32, [None, 1])
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

    def lstm_cell(self, hidden_size):
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        if self.training:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)
        return cell

    def train(self, x_data, y_data):
        self.training = True
        return self.sess.run([self.reg_loss, self.loss, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})

    def predict(self, x_data):
        self.training = False
        return self.sess.run(self.Y_, feed_dict={self.X: x_data})

    def rmse_predict(self, targets, predictions):
        self.training = False
        return self.sess.run(self.rmse, feed_dict={self.targets: targets, self.predictions: predictions})

class CNN_Model:
    def __init__(self, sess, model_name):
        self.sess = sess
        self.model_name = model_name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.model_name):
            with tf.name_scope('input_layer'):
                self.learning_rate = 0.001
                self.training = tf.placeholder(tf.bool, name='training')
                self.regularizer = tf.contrib.layers.l2_regularizer(0.0005)

                self.X = tf.placeholder(dtype=tf.float32, shape=[None, 100])
                X_data = tf.reshape(self.X, [-1, 10, 10, 1])
                self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 100])

            with tf.name_scope('conv_layer'):
                self.W1_conv = tf.get_variable(name='W1_conv', shape=[1, 3, 1, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L1_conv = tf.nn.conv2d(input=X_data, filter=self.W1_conv, strides=[1, 1, 1, 1], padding='VALID')  # 10x10 -> 10x8
                self.L1_conv = self.BN(input=self.L1_conv, training=self.training, name='L1_conv_BN')
                self.L1_conv = self.parametric_relu(self.L1_conv, 'R1_conv')

                self.W2_conv = tf.get_variable(name='W2_conv', shape=[3, 1, 40, 40], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L2_conv = tf.nn.conv2d(input=self.L1_conv, filter=self.W2_conv, strides=[1, 1, 1, 1], padding='VALID')  # 10x8 -> 8x8
                self.L2_conv = self.BN(input=self.L2_conv, training=self.training, name='L2_conv_BN')
                self.L2_conv = self.parametric_relu(self.L2_conv, 'R2_conv')

                self.W3_conv = tf.get_variable(name='W3_conv', shape=[1, 3, 40, 80], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L3_conv = tf.nn.conv2d(input=self.L2_conv, filter=self.W3_conv, strides=[1, 1, 1, 1], padding='VALID')  # 8x8 -> 8x6
                self.L3_conv = self.BN(input=self.L3_conv, training=self.training, name='L3_conv_BN')
                self.L3_conv = self.parametric_relu(self.L3_conv, 'R3_conv')

                self.W4_conv = tf.get_variable(name='W4_conv', shape=[3, 1, 80, 80], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L4_conv = tf.nn.conv2d(input=self.L3_conv, filter=self.W4_conv, strides=[1, 1, 1, 1], padding='VALID')  # 8x6 -> 6x6
                self.L4_conv = self.BN(input=self.L4_conv, training=self.training, name='L4_conv_BN')
                self.L4_conv = self.parametric_relu(self.L4_conv, 'R4_conv')

                self.W5_conv = tf.get_variable(name='W5_conv', shape=[1, 3, 80, 120], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L5_conv = tf.nn.conv2d(input=self.L4_conv, filter=self.W5_conv, strides=[1, 1, 1, 1], padding='VALID')  # 6x6 -> 6x4
                self.L5_conv = self.BN(input=self.L5_conv, training=self.training, name='L5_conv_BN')
                self.L5_conv = self.parametric_relu(self.L5_conv, 'R5_conv')

                self.W6_conv = tf.get_variable(name='W6_conv', shape=[3, 1, 120, 120], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.L6_conv = tf.nn.conv2d(input=self.L5_conv, filter=self.W6_conv, strides=[1, 1, 1, 1], padding='VALID')  # 6x4 -> 4x4
                self.L6_conv = self.BN(input=self.L6_conv, training=self.training, name='L6_conv_BN')
                self.L6_conv = self.parametric_relu(self.L6_conv, 'R6_conv')
                self.L6_conv = tf.reshape(self.L6_conv, [-1, 4 * 4 * 120])

            with tf.name_scope('fc_layer'):
                self.W1_fc = tf.get_variable(name='W1_fc', shape=[4 * 4 * 120, 650], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b1_fc = tf.Variable(tf.constant(value=0.001, shape=[650], name='b1_fc'))
                self.L1_fc = tf.matmul(self.L6_conv, self.W1_fc) + self.b1_fc
                self.L1_fc = self.BN(input=self.L1_fc, training=self.training, name='L1_fc_BN')
                self.L1_fc = self.parametric_relu(self.L1_fc, 'R1_fc')

                self.W2_fc = tf.get_variable(name='W2_fc', shape=[650, 650], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b2_fc = tf.Variable(tf.constant(value=0.001, shape=[650], name='b2_fc'))
                self.L2_fc = tf.matmul(self.L1_fc, self.W2_fc) + self.b2_fc
                self.L2_fc = self.BN(input=self.L2_fc, training=self.training, name='L2_fc_BN')
                self.L2_fc = self.parametric_relu(self.L2_fc, 'R2_fc')

            self.W_out = tf.get_variable(name='W_out', shape=[650, 100], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[100], name='b_out'))
            self.logits = tf.matmul(self.L2_fc, self.W_out) + self.b_out

            self.reg_cost = tf.reduce_sum([self.regularizer(train_var) for train_var in tf.get_variable_scope().trainable_variables() if re.search(self.model_name+'\/W', train_var.name) is not None])
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + 0.0005 * self.reg_cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    def BN(self, input, training, name):
        return tf.contrib.layers.batch_norm(input, decay=0.99, scale=True, is_training=training, updates_collections=None, scope=name)

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

def min_max_scaler(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-5)

def read_data(file_name):
    data = np.loadtxt('data/'+file_name, delimiter=',', skiprows=1)
    data = data[:, 1:]
    data = data[np.sum(np.isnan(data), axis=1) == 0]
    data = min_max_scaler(data)
    x, y = data, data[:, [3]]
    dataX = []
    dataY = []
    for i in range(0, len(data) - n_sequences):
        _x = x[i: i + n_sequences]
        _y = y[i + n_sequences]
        # _y = np.array(y[i+1: i + n_sequences + 1]).flatten().tolist()
        dataX.append(_x)
        dataY.append(_y)
    return dataX, dataY

n_inputs = 7
n_sequences = 10
n_hiddens = 7
n_outputs = 1
hidden_layer_cnt = 3

file_list = os.listdir('data/')
rnn_model_list = []
cnn_model_list = []
cnn_input_size = 10 * 10

batch_size = 100
epochs = 20
step_size = batch_size * cnn_input_size

with tf.Session() as sess:
    for idx, file_name in enumerate(file_list):
        rnn_model_list.append(RNN_Model(sess=sess, n_inputs=n_inputs, n_sequences=n_sequences, n_hiddens=n_hiddens,
                                        n_outputs=n_outputs, hidden_layer_cnt=hidden_layer_cnt, file_name=file_name, model_name='RNN_Model_' + str(idx+1)))

    for idx in range(len(rnn_model_list)):
        cnn_model_list.append(CNN_Model(sess=sess, model_name='CNN_Model_' + str(idx+1)))

    sess.run(tf.global_variables_initializer())

    for rnn_model, cnn_model in zip(rnn_model_list, cnn_model_list):
        total_X, total_Y = read_data(rnn_model.file_name)  # 모델별 파일 로딩
        train_X, train_Y = total_X[:int(len(total_Y)*0.7)], total_Y[:int(len(total_Y)*0.7)]  # train 데이터
        test_X, test_Y = total_X[int(len(total_Y)*0.7):], total_Y[int(len(total_Y)*0.7):]  # test 데이터
        train_len, test_len = len(train_Y), len(test_Y)

        stime = time.time()
        print('training start -')
        print('train data -', train_len, ', test data -', test_len)
        for epoch in range(epochs):
            train_loss = 0.
            pred_data = np.zeros(train_len, dtype=np.float32)
            estime = time.time()
            for idx in range(0, train_len, batch_size):
                sample_size = train_len if batch_size > train_len else batch_size
                batch_X, batch_Y = train_X[idx: idx+sample_size], train_Y[idx: idx+sample_size]
                reg_loss, loss, _ = rnn_model.train(batch_X, batch_Y)
                predicts = rnn_model.predict(batch_X)
                pred_data[idx: idx+sample_size] = np.array(predicts).flatten()
                train_loss += loss / sample_size
                train_len -= sample_size
            eetime = time.time()
            print('RNN Model :', rnn_model.model_name, ', epoch :', epoch+1, ', loss :', train_loss, ' -', eetime-estime)
            train_len, test_len = len(train_Y), len(test_Y)

            train_loss = 0.
            estime = time.time()
            for idx in range(0, len(pred_data) - step_size, step_size):
                if idx + step_size > len(pred_data):
                    sample_size = (idx + step_size) - len(pred_data) - (((idx + step_size) - len(pred_data)) % cnn_input_size)
                else:
                    sample_size = step_size
                batch_X, batch_Y = np.array(pred_data[idx: idx + sample_size]).reshape([int(sample_size/cnn_input_size), cnn_input_size]).tolist(), \
                                   np.array(train_Y[n_sequences + idx: n_sequences + idx + sample_size]).reshape([int(sample_size/cnn_input_size), cnn_input_size]).tolist()
                loss, _ = cnn_model.train(batch_X, batch_Y)
                train_loss += loss / int(sample_size/cnn_input_size)
            eetime = time.time()
            print('CNN Model :', cnn_model.model_name, ', epoch :', epoch + 1, ', loss :', train_loss, ' -', eetime - estime)
        etime = time.time()
        print('training end -', etime-stime, '\n')

        print('testing start -')
        test_rmse = 0.
        pred_data = np.zeros(test_len, dtype=np.float32)
        for idx in range(0, test_len, batch_size):
            sample_size = test_len if batch_size > test_len else batch_size
            batch_X, batch_Y = test_X[idx: idx + sample_size], test_Y[idx: idx + sample_size]
            predicts = rnn_model.predict(batch_X)
            rmse = rnn_model.rmse_predict(batch_Y, predicts)
            test_rmse += rmse / sample_size
            test_len -= sample_size
        print('RNN Model :', rnn_model.model_name, ', rmse :', test_rmse)

        cnt = 0
        test_loss = 0.
        accuracy = 0.
        estime = time.time()
        final_predicts = []
        final_y = []
        for idx in range(0, len(pred_data) - step_size, step_size):
            if idx + step_size > len(pred_data):
                sample_size = (idx + step_size) - len(pred_data) - (((idx + step_size) - len(pred_data)) % cnn_input_size)
            else:
                sample_size = step_size
            batch_X, batch_Y = np.array(pred_data[idx: idx + sample_size]).reshape([int(sample_size / cnn_input_size), cnn_input_size]).tolist(), \
                               np.array(test_Y[n_sequences + idx: n_sequences + idx + sample_size]).reshape([int(sample_size / cnn_input_size), cnn_input_size]).tolist()
            predicts = cnn_model.predict(batch_X)
            final_predicts += np.array(predicts).flatten().tolist()
            final_y += np.array(batch_Y).flatten().tolist()
            loss, _ = cnn_model.train(batch_X, batch_Y)
            test_loss += loss / int(sample_size / cnn_input_size)
        eetime = time.time()
        print('CNN Model :', cnn_model.model_name, ', loss :', test_loss, ' -', eetime - estime)
        print('testing end -')

        # Plot predictions
        plt.plot(final_y)
        plt.plot(final_predicts)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.show()