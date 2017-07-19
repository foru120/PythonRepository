import tensorflow as tf
import numpy as np

sentence = "if you want to build a ship, don't drum up people together to " \
           "collect wood and don't assign them tasks and work, but rather " \
           "teach them to long for the endless immensity of the sea."
seq_length = 7
num_classes = 7
hidden_size = 100
batch_size = 1
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

dataX = []
dataY = []

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i: i + seq_length]
    y_str = sentence[i+1: i + seq_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape

# Make a lstm cell with hidden_size (each unit output vector size)
cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell]*2, state_is_tuple=True)  # 각 층에 대한 hidden_size 를 다르게 할 경우 여러개의 BasicLSTMCell 을 만들어서 넣는다.

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)

X_for_softmax = tf.reshape(outputs, [-1, hidden_size])
softmax_w = tf.get_variable('softmax_w', [hidden_size, num_classes])
softmax_b = tf.get_variable('softmax_b', [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])

weights = tf.ones([batch_size, seq_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
