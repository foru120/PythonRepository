import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = 300
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

with tf.contrib.framework.arg_scope(
    [fully_connected],
    activation_fn = tf.nn.elu,
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
    weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)):

    hidden1 = fully_connected(X, n_hidden1)
    hidden2 = fully_connected(hidden1, n_hidden2)
    hidden3 = fully_connected(hidden2, n_hidden3)
    outputs = fully_connected(hidden3, n_outputs, activation_fn=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 5
batch_size = 150
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            l, _ = sess.run([loss, training_op], feed_dict={X: X_batch})
            print(l)