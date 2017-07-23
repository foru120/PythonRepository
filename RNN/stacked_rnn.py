import tensorflow as tf
import numpy as np

sentence = "if you want to build a ship, don't drum up people together to " \
           "collect wood and don't assign them tasks and work, but rather " \
           "teach them to long for the endless immensity of the sea."

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)  # 입력 문자 하나당 차원의 개수
hidden_size = len(char_set)  # RNN Cell 출력 개수
num_classes = len(char_set)  # 최종 출력 class 개수
sequence_length = 10
learning_rate = 0.1

dataX = []
dataY = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i: i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

###################################################################################################
## ▣ tf.contrib.rnn.BasicLSTMCell
##  - num_units      : LSTM cell 의 단위 개수
##  - state_is_tuple : True 이면 c_state 와 m_state 의 2개의 tuple 로 리턴이되고, False 이면 column 축 사이가 연결되어 리턴
##  - activation     : 활성화 함수 (default : tanh)
##  - reuse          : 기존 scope 에서 변수들을 재사용하기 위한 옵션
###################################################################################################
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # training
    for i in range(500):
        _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)  # 각 글자별 class 중 최대값의 인덱스를 리턴
            print(i, j, ''.join([char_set[t] for t in index]), l)

    # test
    results = sess.run(outputs, feed_dict={X: dataX})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')