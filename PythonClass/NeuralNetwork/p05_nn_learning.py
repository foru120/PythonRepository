from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

training_epochs = 2
learning_rate = 0.01
batch_size = 100

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.variable_scope('test'):
    with tf.name_scope('hidden_layer') as scope:
        W1 = tf.Variable(0.01 * tf.random_normal(shape=[784, 50]), name='weight1')
        b1 = tf.Variable(tf.constant(value=0.01, shape=[50]), name='bias1')
        L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

        W1_hist = tf.summary.histogram('weights1', W1)
        b1_hist = tf.summary.histogram('bias1', b1)
        L1_hist = tf.summary.histogram('layer1', L1)

    with tf.name_scope('output_layer') as scope:
        W2 = tf.Variable(0.01 * tf.random_normal(shape=[50, 10]), name='weight2')
        b2 = tf.Variable(tf.constant(value=0.01, shape=[10]), name='bias2')
        y_ = tf.nn.softmax(tf.matmul(L1, W2) + b2)

        W2_hist = tf.summary.histogram('weight2', W2)
        b2_hist = tf.summary.histogram('bias2', b2)
        y_hist = tf.summary.histogram('y_', y_)

loss = -tf.reduce_sum(Y * tf.log(y_))
loss_hist = tf.summary.scalar('loss', loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
accuracy_hist = tf.summary.scalar('accuracy', accuracy)

summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./logs/nn_learning')
    writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            s, l, _ = sess.run([summary, loss, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            writer.add_summary(s, global_step=epoch+i)
            avg_cost += l / total_batch
        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Learning Finished!')

    test_batch = mnist.test.next_batch(1000)
    print('test accuracy %g' % sess.run(accuracy, feed_dict={X: test_batch[0], Y: test_batch[1]}))