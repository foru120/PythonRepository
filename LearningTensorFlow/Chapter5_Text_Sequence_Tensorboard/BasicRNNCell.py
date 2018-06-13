import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\\Source\\PythonRepository\\LearningTensorFlow\\Chapter5_Text_Sequence_Tensorboard\\mnist_data', one_hot=True)

element_size = 28; time_steps = 28; num_classes = 10
batch_size = 128; hidden_layer_size = 128

_inputs = tf.placeholder(tf.float32, shape=[None,
                                            time_steps,
                                            element_size],
                         name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='inputs')

# 텐서플로의 기본 제공 함수들
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=_inputs, dtype=tf.float32)

last_rnn_output = tf.squeeze(outputs[:, -1, :])

final_output = tf.layers.dense(inputs=last_rnn_output, units=10)

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

for i in range(3001):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, time_steps, element_size))
    sess.run(train_step, feed_dict={_inputs: batch_x, y: batch_y})

    if i % 1000 == 0:
        acc = sess.run(accuracy, feed_dict={_inputs: batch_x, y: batch_y})