import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 랜덤시트 고정시키기
np.random.seed(5)

dataset = np.loadtxt('./dataset/diabetes.csv', delimiter=',', skiprows=1)

# 데이터셋 생성하기
x_train = dataset[:700, 0:8]
y_train = dataset[:700, 8]
x_test = dataset[700:, 0:8]
y_test = dataset[700:, 8]

# 모델 구성하기
model = Sequential()
model.add(Dense(units=12, input_dim=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습시키기
model.fit(x_train, y_train, epochs=1500, batch_size=64)

# 모델 평가하기
scores = model.evaluate(x_test, y_test)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1]*100))