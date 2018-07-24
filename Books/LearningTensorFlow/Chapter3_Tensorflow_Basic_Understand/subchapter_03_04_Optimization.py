import tensorflow as tf

x = tf.placeholder(dtype=tf.float64, shape=[None, 3])
y_true = tf.placeholder(dtype=tf.float64, shape=None)

w = tf.Variable([[0, 0, 0]], dtype=tf.float64, name='weights')
b = tf.Variable(0, dtype=tf.float64, name='bias')

y_pred = tf.matmul(w, tf.transpose(x)) + b

"""MSE(Mean Square Error)"""
loss = tf.reduce_mean(tf.square(y_true - y_pred))

"""Cross Entropy"""
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
loss = tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)