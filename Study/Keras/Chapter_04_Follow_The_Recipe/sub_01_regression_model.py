# 1. 데이터셋 준비
import numpy as np
from sklearn.metrics import mean_squared_error
import random

# 데이터셋 생성
x_train = np.random.random((1000, 1))
y_train = x_train * 2 + np.random.random((1000, 1)) / 3.0

x_test = np.random.random((100, 1))
y_test = x_test * 2 + np.random.random((100, 1)) / 3.0

x_train = x_train.reshape(1000,)
y_train = y_train.reshape(1000,)
x_test = x_test.reshape(100,)
y_test = y_test.reshape(100,)

# 데이터셋 확인
# import matplotlib.pyplot as plt
#
# plt.plot(x_train, y_train, 'ro')
# plt.plot(x_test, y_test, 'bo')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

w = np.cov(x_train, y_train, bias=1)[0, 1] / np.var(x_train)
b = np.average(y_train) - w * np.average(x_train)

print(w, b)

# 3. 모델 평가하기
y_predict = w * x_test + b
mse = mean_squared_error(y_test, y_predict)
print('mse : ' + str(mse))