import os
import numpy as np
import tensorflow as tf
import time
import re
import matplotlib.pyplot as plt
import copy as cp

class NN_Model:
    def __init__(self, sess, model_name, n_sequences, n_inputs):
        self.sess = sess
        self.model_name = model_name
        self.n_sequences = n_sequences
        self.n_inputs = n_inputs
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.model_name):
            with tf.name_scope('input_layer'):
                self.learning_rate = 0.001
                self.training = tf.placeholder(tf.bool, name='training')
                self.regularizer = tf.contrib.layers.l2_regularizer(0.0005)

                self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_sequences, self.n_inputs])
                X_data = tf.reshape(self.X, [-1, self.n_sequences * self.n_inputs])
                self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])  ##############3

                # 이렇게 했더니 (원본 model이 살아있어서일 것) adam optimizer돌릴 때 중복된 이름이라 안 된다고 뜸
                # self.b1_pre = tf.get_default_graph().get_tensor_by_name('Stacked_Autoencoder_Model_1/fc_layer/Variable:0')
                # self.W2_pre = tf.get_default_graph().get_tensor_by_name('Stacked_Autoencoder_Model_1/W2_fc:0')
                # self.b2_pre = tf.get_default_graph().get_tensor_by_name('Stacked_Autoencoder_Model_1/fc_layer/Variable_1:0')

                # assign은 tf.Variable일 때만 제대로 assign된다??
                # Stacked_Autoencoder_Model_1/W1_fc:0  이 np.ndarray형태이기 때문인지 assign되지 않음
                # self.W1_pre = tf.assign(tf.get_default_graph().get_tensor_by_name('Stacked_Autoencoder_Model_1/W1_fc:0'),self.W1_pre)
                # self.W2_pre = tf.assign(tf.get_default_graph().get_tensor_by_name('Stacked_Autoencoder_Model_1/W1_fc:0'), self.W2_pre)
                # self.b1_pre = tf.assign(tf.get_default_graph().get_tensor_by_name('Stacked_Autoencoder_Model_1/W1_fc:0'), self.b1_pre)
                # self.b1_pre = tf.assign(tf.get_default_graph().get_tensor_by_name('Stacked_Autoencoder_Model_1/W1_fc:0'), self.b2_pre)

                self.W1_pre = cp.deepcopy(sess.run('Stacked_Autoencoder_Model_1/W1_fc:0'))
                self.W2_pre = cp.deepcopy(sess.run('Stacked_Autoencoder_Model_1/W2_fc:0'))
                self.b1_pre = cp.deepcopy(sess.run('Stacked_Autoencoder_Model_1/fc_layer/Variable:0'))
                self.b2_pre = cp.deepcopy(sess.run('Stacked_Autoencoder_Model_1/fc_layer/Variable_1:0'))

            with tf.name_scope('DNN'):
                # self.W1_pre = sess.run('Stacked_Autoencoder_Model_1/W1_fc:0')  # uninitialized error?
                # self.b1_pre = sess.run('Stacked_Autoencoder_Model_1/fc_layer/Variable:0')
                self.L1_fc = tf.matmul(X_data, self.W1_pre) + self.b1_pre
                self.L1_fc = self.BN(input=self.L1_fc, training=self.training, name='L1_fc_BN')
                self.L1_fc = self.parametric_relu(self.L1_fc, 'L1_fc')

                # self.W2_pre = sess.run('Stacked_Autoencoder_Model_1/W2_fc:0')
                # self.b2_pre = sess.run('Stacked_Autoencoder_Model_1/fc_layer/Variable_1:0')
                self.L2_fc = tf.matmul(self.L1_fc, self.W2_pre) + self.b2_pre
                self.L2_fc = self.BN(input=self.L2_fc, training=self.training, name='L2_fc_BN')
                self.L2_fc = self.parametric_relu(self.L2_fc, 'L2_fc')

                self.W3_fc = tf.get_variable(name='W3_fc', shape=[448, 448], dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b3_fc = tf.Variable(tf.constant(value=0.001, shape=[448], name='b3_fc'))
                self.L3_fc = tf.matmul(self.L2_fc, self.W3_fc) + self.b3_fc
                self.L3_fc = self.BN(input=self.L3_fc, training=self.training, name='L3_fc_BN')
                self.L3_fc = self.parametric_relu(self.L3_fc, 'R3_fc')

                self.W4_fc = tf.get_variable(name='W4_fc', shape=[448, 448], dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                self.b4_fc = tf.Variable(tf.constant(value=0.001, shape=[448], name='b4_fc'))
                self.L4_fc = tf.matmul(self.L3_fc, self.W4_fc) + self.b4_fc
                self.L4_fc = self.BN(input=self.L4_fc, training=self.training, name='L4_fc_BN')
                self.L4_fc = self.parametric_relu(self.L4_fc, 'R4_fc')

                # self.W2_fc = tf.get_variable(name='W2_fc', shape=[300, 300], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
                # self.b2_fc = tf.Variable(tf.constant(value=0.001, shape=[300], name='b2_fc'))
                # self.L2_fc = tf.matmul(self.L1_fc, self.W2_fc) + self.b2_fc
                # self.L2_fc = self.BN(input=self.L2_fc, training=self.training, name='L2_fc_BN')
                # self.L2_fc = self.parametric_relu(self.L2_fc, 'R2_fc')

            self.W_out = tf.get_variable(name='W_out', shape=[448, 1], dtype=tf.float32,
                                         initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b_out = tf.Variable(tf.constant(value=0.001, shape=[1], name='b_out'))
            self.logits = tf.matmul(self.L3_fc, self.W_out) + self.b_out

            self.reg_cost = tf.reduce_sum(
                [self.regularizer(train_var) for train_var in tf.get_variable_scope().trainable_variables() if
                 re.search(self.model_name + '\/W', train_var.name) is not None])
            self.cost = tf.reduce_sum(tf.square(self.logits - self.Y)) + 0.0005 * self.reg_cost
            # self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + 0.0005 * self.reg_cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.cost)

    def BN(self, input, training, name):
        return tf.contrib.layers.batch_norm(input, decay=0.99, scale=True, is_training=training,
                                            updates_collections=None, scope=name)

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    def test_loss(self, predicts, y_test):
        self.loss = tf.reduce_mean(tf.square(predicts - self.Y))
        return self.sess.run(self.loss, feed_dict={self.Y: y_test, self.training: False})

n_inputs = 7
n_sequences = 256
n_hiddens = 7
n_outputs = 1
hidden_layer_cnt = 5

def min_max_scaler(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-5)

def read_data(file_name, n_sequences):
    data = np.loadtxt('C:\\bitcoin/' + file_name, delimiter=',', skiprows=1)
    data = data[:, 1:]
    data = data[np.sum(np.isnan(data), axis=1) == 0]
    data = min_max_scaler(data)
    x, y = data, data[:, [3]]
    dataX = []
    dataY = []
    for i in range(0, len(data) - n_sequences):
        _x = x[i:i + n_sequences]
        _y = y[i + n_sequences]
        dataX.append(_x)
        dataY.append(_y)
    return dataX, dataY

file_list = os.listdir('C:\\bitcoin/')
model_list = []
batch_size = 100
encoder_epoch = 30
epochs = 10

checkpoint_file = "C:\weight\stacked_autoencoder_weight12.chk"
idx = 0

for file_name in file_list:
    total_X, total_Y = read_data(file_name, n_sequences)  # 모델별 파일 로딩
    train_X, train_Y = total_X[:int(len(total_Y) * 0.7)], total_Y[:int(len(total_Y) * 0.7)]  # train 데이터
    test_X, test_Y = total_X[int(len(total_Y) * 0.7):], total_Y[int(len(total_Y) * 0.7):]  # test 데이터
    train_len, test_len = len(train_Y), len(test_Y)

    with tf.Session() as sess:  # CNN 부분
        # for idx, file_name in enumerate(file_list):
        # model_list.append(Model(sess=sess, n_inputs=n_inputs, n_sequences=n_sequences, n_hiddens=n_hiddens,
        #                        n_outputs=n_outputs, hidden_layer_cnt=hidden_layer_cnt, file_name=file_name,
        #                        model_name='Model_' + str(idx + 1)))
        # awdksldk = tf.constant(1)
        saver = tf.train.import_meta_graph('C:\weight\stacked_autoencoder_weight12.chk.meta')
        saver.restore(sess, checkpoint_file)  # 이 세션에 stacked autoencoder에서 W1,W2 불러오기
        model = NN_Model(sess=sess, model_name='Model_' + file_name, n_sequences=n_sequences, n_inputs=n_inputs)
        sess.run(tf.global_variables_initializer())

        stime = time.time()
        print(model.model_name, ', training start -')
        print('train data -', train_len, ', test data -', test_len)

        for epoch in range(epochs):
            train_loss = 0.
            for idx in range(0, train_len, batch_size):
                sample_size = train_len if batch_size > train_len else batch_size
                batch_X, batch_Y = train_X[idx: idx + sample_size], train_Y[idx: idx + sample_size]
                loss, _ = model.train(batch_X, batch_Y)
                train_loss += loss / sample_size
                train_len -= sample_size
            print('Model :', model.model_name, ', epoch :', epoch + 1, ', loss :', train_loss)
            train_len, test_len = len(train_Y), len(test_Y)
        print(model.model_name, ', training end -\n')

        print(model.model_name, ', testing start -')
        test_loss = 0.
        final_predicts = []
        final_y = []
        for idx in range(0, test_len, batch_size):
            sample_size = test_len if batch_size > test_len else batch_size
            batch_X, batch_Y = test_X[idx: idx + sample_size], test_Y[idx: idx + sample_size]
            predicts = model.predict(batch_X)
            # rmse = model.rmse_predict(batch_Y, predicts)
            # test_rmse += rmse / sample_size
            test_loss += model.test_loss(predicts, batch_Y)
            test_len -= sample_size
            final_y += np.array(batch_Y).flatten().tolist()
            final_predicts += np.array(predicts).flatten().tolist()
        etime = time.time()
        print('Model :', model.model_name, ', test_loss :', test_loss)
        print(model.model_name, ', testing end -')
        print(model.model_name, ', time -', etime - stime, '\n')

        idx += 1  # 모델별 idx
        # Plot predictions
        plt.plot(final_y, ls="--", marker="o", label='y')
        plt.plot(final_predicts, ls="--", marker="o", label='predict')
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.legend(loc=1)
        plt.show()