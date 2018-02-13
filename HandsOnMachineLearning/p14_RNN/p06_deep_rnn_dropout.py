import tensorflow as tf
import sys

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1
n_layers = 5
n_iterations = 100

keep_prob = 0.5

is_training = (sys.argv[-1] == 'train')

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_steps, n_neurons])

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
lstm_cell1 = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True)
gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)

if is_training:
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)  # RNN 의 각 계층마다 드롭아웃을 구현
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    if is_training:
        init.run()
        for iteration in range(n_iterations):
            # train the model
            pass
        save_path = saver.save(sess, 'train_log/my_model.ckpt')
    else:
        saver.restore(sess, 'train_log/my_model.ckpt')
        # use the model