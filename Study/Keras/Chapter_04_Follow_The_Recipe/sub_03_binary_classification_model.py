import numpy as np

# 데이터셋 생성
# x_train = np.random.random((1000, 12))
# y_train = np.random.randint(2, size=(1000, 1))
# x_test = np.random.random((100, 12))
# y_test = np.random.randint(2, size=(100, 1))

# 데이터셋 확인 (2차원)
# import matplotlib.pyplot as plt
#
# plot_x = x_train[:,0]
# plot_y = x_train[:,1]
# plot_color = y_train.reshape(1000,)
#
# plt.scatter(plot_x, plot_y, c=plot_color)
# plt.show()

# 데이터셋 확인 (3차원)
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# plot_x = x_train[:,0]
# plot_y = x_train[:,1]
# plot_z = x_train[:,2]
# plot_color = y_train.reshape(1000,)
#
# ax.scatter(plot_x, plot_y, plot_z, c=plot_color)
# plt.show()

# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random

# 1. 데이터셋 생성하기
x_train = np.random.random((1000, 12))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 12))
y_test = np.random.randint(2, size=(100, 1))

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

# 5. 학습과정 살펴보기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('loss_and_metrics : ' + str(loss_and_metrics))