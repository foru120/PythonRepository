########################################################################################################################
## ▣ recurrent neural network
##  - 기존에 사용된 neural network 에서 hidden layer 에 recurrent 한 연결이 있는 신경망 구조.
##  - 총 3개의 Weight 이 존재. (W(xh), W(hh), W(hy))
##  ■ 활용가능 분야
##   - Language modeling
##   - Speech recognition
##   - Machine Translation
##   - Coversation Modeling / Question Answering
##   - Image / Video captioning
##   - Image / Music / Dance generation
########################################################################################################################

import tensorflow as tf
import numpy as np

idx2char = ['h', 'i', 'e', 'l', 'o']
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]
y_data = [[1, 0, 2, 3, 3, 4]]

sequence_length = 6
input_dims = 5
hidden_size = 5
batch_size = 1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dims])
Y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, 'loss: ', l, 'prediction: ', result, 'true Y: ', y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print('\tPrediction str: ', ''.join(result_str))