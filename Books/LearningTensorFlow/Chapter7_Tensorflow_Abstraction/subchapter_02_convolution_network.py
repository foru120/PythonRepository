import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#todo tensorflow.contrib.layers 를 이용한 CNN 신경망 생성
def model_fn(x, target, mode, params):
    x_image = tf.reshape(x, shape=[-1, 28, 28, 1])
    y_ = tf.cast(target, tf.float32)

    conv1 = layers.convolution2d(inputs=x_image, num_outputs=32, kernel_size=[5, 5], activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.1),
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pool1 = layers.max_pool2d(conv1, [2, 2])

    conv2 = layers.convolution2d(inputs=pool1, num_outputs=64, kernel_size=[5, 5], activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.1),
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pool2 = layers.max_pool2d(conv2, [2, 2])

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = layers.fully_connected(inputs=pool2_flat, num_outputs=1024,
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.1),
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc1_drop = layers.dropout(inputs=fc1, keep_prob=params['dropout'], is_training=(mode == 'train'))

    y_conv = layers.fully_connected(inputs=fc1_drop, num_outputs=10, activation_fn=None)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params['learning_rate'],
        optimizer='Adam'
    )

    predictions = tf.argmax(y_conv, 1)

    return predictions, cross_entropy, train_op

DATA_DIR = './mnist_data'

data = input_data.read_data_sets(train_dir=DATA_DIR, one_hot=True)
x_data, y_data = data.train.images, data.train.labels.astype(np.int32)
tf.cast(x_data, tf.float32)
tf.cast(y_data, tf.float32)

model_params = {'learning_rate': 1e-4, 'dropout': 0.5}

CNN = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    params=model_params
)

print('Starting training for %s steps max' % 5000)

CNN.fit(x=x_data,
        y=y_data,
        batch_size=50,
        max_steps=5000)

test_acc = 0

for ii in range(5):
    batch = data.test.next_batch(2000)
    predictions = list(CNN.predict(x=batch[0], as_iterable=True))
    test_acc += (np.argmax(batch[1], 1) == predictions).mean()

print(test_acc/5)