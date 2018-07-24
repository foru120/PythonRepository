import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np

#todo 데이터를 로딩하고 기본적인 변환을 수행
import tflearn.datasets.mnist as mnist

X, Y, X_test, Y_test = mnist.load_data('./mnist_data', one_hot=True)
X = X.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])

#todo 네트워크 구성
CNN = input_data(shape=[None, 28, 28, 1], name='input')
CNN = conv_2d(incoming=CNN, nb_filter=32, filter_size=5, activation='relu', regularizer='L2')
CNN = max_pool_2d(incoming=CNN, kernel_size=2)
CNN = local_response_normalization(incoming=CNN)
CNN = conv_2d(incoming=CNN, nb_filter=64, filter_size=5, activation='relu', regularizer='L2')
CNN = max_pool_2d(incoming=CNN, kernel_size=2)
CNN = local_response_normalization(incoming=CNN)
CNN = fully_connected(incoming=CNN, n_units=1024, activation=None)
CNN = dropout(incoming=CNN, keep_prob=0.5)
CNN = fully_connected(incoming=CNN, n_units=10, activation='softmax')
CNN = regression(incoming=CNN, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='target')

#todo 네트워크 학습
model = tflearn.DNN(network=CNN, tensorboard_verbose=0, tensorboard_dir='./MNIST_tflearn_board/cnn',
                    checkpoint_path='./MNIST_tflearn_checkpoints/cnn')
model.fit({'input': X}, {'target': Y}, n_epoch=3, validation_set=({'input': X_test}, {'target': Y_test}),
          snapshot_step=1000, show_metric=True, run_id='convnet_mnist')

#todo 성능 평가
evaluation = model.evaluate({'input': X_test}, {'target': Y_test})
print(evaluation)

#todo 모델 예측
pred = model.predict({'input': X_test})
print((np.argmax(Y_test, 1) == np.argmax(pred, 1)).mean())
