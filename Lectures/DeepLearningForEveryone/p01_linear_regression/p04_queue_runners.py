import tensorflow as tf

filename_queue = tf.train.string_input_producer(['data/data-01-test-score.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# shuffle_batch
batch_size = 10
min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
train_x_batch, train_y_batch = tf.train.shuffle_batch([xy[0:-1], xy[-1:]], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

X = tf.placeholder(tf.float32, shape=[None, 3], name='x_data')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='y_data')

W = tf.Variable(tf.random_normal(shape=[3, 1]), name='weight')
b = tf.Variable(tf.random_normal(shape=[1]), name='bias')

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, h_val, _ = sess.run([cost, H, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 100 == 0:
        print(step, 'Cost : ', cost_val, '\nPrediction : ', h_val)

coord.request_stop()
coord.join(threads)