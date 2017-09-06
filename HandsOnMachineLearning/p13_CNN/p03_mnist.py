from HandsOnMachineLearning.p13_CNN.setup import *
from tensorflow.examples.tutorials.mnist import input_data

height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = 'SAME'

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = 'SAME'

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

reset_graph()

with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name='y')

conv1 = tf.layers.conv2d(X_reshaped, filters=conv2_fmaps, kernel_size=conv1_ksize, strides=conv1_stride, padding=conv1_pad)
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride, padding=conv2_pad)

with tf.name_scope('pool3'):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

with tf.name_scope('fc1'):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name='fc1')

with tf.name_scope('output'):
    logits = tf.layers.dense(fc1, n_outputs, name='output')  # activation=None : tf.matmul() + tf.add() 연산만 수행
    Y_proba = tf.nn.softmax(logits, name='Y_proba')

with tf.name_scope('train'):
    # cross-entropy : 무엇이 정말 참인지를 고려해 볼 때, neural network 의 예측을 믿는 것이 얼마나 나쁜지를 설명하는 이론
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)  # y 값을 자동적으로 one-hot encoding 으로 변환
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)  # predict 시에 top-5, top-3 등을 예측할 때 사용
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

mnist = input_data.read_data_sets('data/')

n_epochs = 10
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, 'Train accuracy: ', acc_train, 'Test accuracy: ', acc_test)

        save_path = saver.save(sess, 'logs/my_mnist_model')