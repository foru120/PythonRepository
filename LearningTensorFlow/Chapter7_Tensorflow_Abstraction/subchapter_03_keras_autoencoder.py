from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
import numpy as np

#todo keras 를 통한 cifar10 데이터 로드
'''cifar10.load_data -> ~/.keras/datasets/ 디렉토리에 다운로드 수행'''
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[np.where(y_train == 1)[0], :, :, :]
x_test = x_test[np.where(y_test == 1)[0], :, :, :]

#todo data normalization
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#todo gaussian noise 추가
x_train_n = x_train + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_train.shape)
x_test_n = x_test + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_test.shape)

x_train_n = np.clip(x_train_n, 0., 1.)
x_test_n = np.clip(x_test_n, 0., 1.)

#todo 함수형 keras 모델 구축
inp_img = Input(shape=(32, 32, 3))
img = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inp_img)
img = MaxPool2D(pool_size=(2, 2), padding='same')(img)
img = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(img)
img = UpSampling2D(size=(2, 2))(img)
decoded = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(img)

autoencoder = Model(inp_img, decoded)

#todo loss, optimizer 정의 및 모델 컴파일
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

tensorboard = TensorBoard(log_dir='./Cifar10_keras_board/autoencoder', histogram_freq=0, write_graph=True, write_images=True)
model_saver = ModelCheckpoint(filepath='./Cifar10_keras_ckpt/autoencoder', verbose=0, period=2)
autoencoder.fit(x=x_train_n, y=x_train_n, epochs=10, batch_size=64, shuffle=True, validation_data=(x_test_n, x_test_n),
                callbacks=[tensorboard, model_saver])
