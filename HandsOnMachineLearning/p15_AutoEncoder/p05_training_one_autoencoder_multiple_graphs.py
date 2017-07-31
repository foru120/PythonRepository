import tensorflow as tf
import sys
import numpy.random as rnd
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

from functools import partial

def train_autoencoder(X_train, n_neurons, n_epochs, batch_size, learning_rate=0.01, l2_reg=0.0005, activation=tf.nn.elu, seed=42):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)
        n_inputs = X_train.shape[1]  # 28*28
        X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        my_dense_layer = partial(  # partial : 함수의 특정 인자값에 대해 초기화가 되어있는 객체를 생성해주는 함수
            tf.layers.dense,
            activation=activation,  # 기본값은 elu 함수
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  # 기본값은 He Initialization
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))  # L2 regularization
        hidden = my_dense_layer(X, n_neurons, name='hidden')  # hidden layer 1
        outputs = my_dense_layer(hidden, n_inputs, activation=None, name='outputs')  # output layer

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE (Mean Square Error)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  # L2 regularization Error
        loss = reconstruction_loss + reg_loss  # MSE + L2 regularization

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = len(X_train) // batch_size
            for iteration in range(n_batches):
                print('\r{}%'.format(100 * iteration // n_batches), end='')
                sys.stdout.flush()
                indices = rnd.permutation(len(X_train))[:batch_size]  # 0 ~ len(X_train) 사이의 숫자를 랜덤으로 생성한 뒤 0 ~ batch_size 사이의 값을 가져옴
                X_batch = X_train[indices]  # 랜덤으로 X_train 값을 추출
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})  # epoch 단위로 loss 를 구함
            print('\r{}'.format(epoch), 'Train MSE:', loss_train)
        params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        hidden_val = hidden.eval(feed_dict={X: X_train})
        return hidden_val, params['hidden/kernel:0'], params['hidden/bias:0'], params['outputs/kernel:0'], params['outputs/bias:0']

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

hidden_output, W1, b1, W4, b4 = train_autoencoder(mnist.train.images, n_neurons=300, n_epochs=4, batch_size=150)
_, W2, b2, W3, b3 = train_autoencoder(hidden_output, n_neurons=150, n_epochs=4, batch_size=150)

reset_graph()

n_inputs = 28*28

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = tf.nn.elu(tf.matmul(X, W1) + b1)
hidden2 = tf.nn.elu(tf.matmul(hidden1, W2) + b2)
hidden3 = tf.nn.elu(tf.matmul(hidden2, W3) + b3)
outputs = tf.matmul(hidden3, W4) + b4

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        # if model_path:
        #     saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

show_reconstructed_digits(X, outputs, model_path='logs/multiple_graphs.ckpt')