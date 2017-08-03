import os
import numpy as np
import tensorflow as tf

class Model:
    def __init__(self, sess, n_inputs, n_sequences, n_hiddens, n_outputs, hidden_layer_cnt, file_name, model_name):
        self.sess = sess
        self.n_inputs = n_inputs
        self.n_sequences = n_sequences
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.hidden_layer_cnt = hidden_layer_cnt
        self.file_name = file_name
        self.model_name = model_name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.model_name):
            self.learning_rate = 0.001

            self.X = tf.placeholder(tf.float32, [None, self.n_sequences, self.n_inputs])
            self.Y = tf.placeholder(tf.float32, [None, self.n_outputs])

            self.multi_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=self.n_hiddens, state_is_tuple=True) for _ in range(self.hidden_layer_cnt)], state_is_tuple=True)
            self.outputs, _states = tf.nn.dynamic_rnn(self.multi_cells, self.X, dtype=tf.float32)
            self.Y_pred = tf.contrib.layers.fully_connected(self.outputs[:, -1], self.n_outputs, activation_fn=None)

            self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.targets = tf.placeholder(tf.float32, [None, 1])
            self.predictions = tf.placeholder(tf.float32, [None, 1])
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

    def train(self, x_data, y_data):
        return self.sess.run([self.loss, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})

    def predict(self, x_data):
        return self.sess.run(self.Y_pred, feed_dict={self.X: x_data})

    def rmse(self, targets, predictions):
        return self.sess.run(self.rmse, feed_dict={self.targets: targets, self.predictions: predictions})

def min_max_scaler(data):
    return (data - np.min(data, axis=0))/(np.max(data, axis=0) - np.min(data, axis=0) + 1e-5)

def read_data(file_name):
    data = np.loadtxt('data/'+file_name, delimiter=',', skiprows=1)
    data = data[:, 1:]
    data = data[np.sum(np.isnan(data), axis=1) == 0]
    data = min_max_scaler(data)
    return data, data[:, [3]]

file_list = os.listdir('data/')
model_list = []

batch_size = 100
epochs = 20

with tf.Session() as sess:
    for idx, file_name in enumerate(file_list):
        model_list.append(Model(sess=sess, n_inputs=7, n_sequences=10, n_hiddens=100, n_outputs=1, hidden_layer_cnt=5, file_name=file_name, model_name='Model_'+str(idx+1)))

    sess.run(tf.global_variables_initializer())

    for model in model_list:
        total_X, total_Y = read_data(model.file_name)  # 모델별 파일 로딩
        train_X, train_Y = total_X[:int(len(total_Y)*0.7)], total_Y[:int(len(total_Y)*0.7)]  # train 데이터
        test_X, test_Y = total_X[int(len(total_Y)*0.7):], total_Y[int(len(total_Y)*0.7):]  # test 데이터
        train_len, test_len = len(train_Y), len(test_Y)

        print(model, ', training start -')
        for epoch in range(epochs):
            train_loss = 0.
            for idx in range(0, train_len, batch_size):
                sample_size = train_len if idx + batch_size > train_len else batch_size
                if sample_size < batch_size:
                    break
                batch_X, batch_Y = train_X[idx: idx+sample_size], train_Y[idx: idx+sample_size]
                loss, _ = model.train(batch_X, batch_Y)
                train_loss += loss / sample_size
                train_len -= sample_size
            print('Model :', model.model_name, ', epoch :', epoch, ', loss :', train_loss)
        print(model, ', training end -\n')

        print(model, ', testing start -')
        test_rmse = 0.
        for idx in range(0, train_len, batch_size):
            sample_size = test_len if idx + batch_size > test_len else batch_size
            if sample_size < batch_size:
                break
            batch_X, batch_Y = test_X[idx: idx + sample_size], test_Y[idx: idx + sample_size]
            predicts = model.predict(batch_X)
            rmse = model.rmse(batch_Y, predicts)
            test_rmse += rmse / sample_size
            test_len -= sample_size
        print('Model :', model.model_name, ', rmse :', test_rmse)
        print(model, ', testing end -\n')