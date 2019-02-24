#-*-coding: utf-8-*-
#todo p.341 ~ p.345
#todo code x ~ code x
#todo 7.3.1 고급 구조 패턴

# ▣ Batch Normalization
# - 훈련하는 동안 평균과 분산이 바뀌더라도 해당 배치에서의 평균과 분산을 통해 정규화하는 방법
# - 훈련 과정에 사용된 배치 데이터의 평균과 분산에 대한 지수 이동 평균을 내부에 유지하고, 테스트 시에 사용
# - Batch Normalization 은 일반적으로 합성곱이나 완전 연결 층 다음에 사용

from keras import layers
from keras.models import Sequential

conv_model = Sequential()
conv_model.add(layers.Conv2D(32, 3, activation='relu'))
conv_model.add(layers.BatchNormalization())

conv_model.add(layers.Dense(32, activation='relu'))
conv_model.add(layers.BatchNormalization())

# ▣ 깊이별 분리 합성곱(Depthwise Separable Convolution)
# - 입력 채널별로 공간 방향의 합성곱을 수행하고 pointwise 합성곱을 통해 출력 채널을 합치는 합성곱
# - 입력에서 공간상 위치는 상관관계가 크지만 채널별로는 매우 독립적이라고 가정하는 경우 최적의 방법

model = Sequential()
model.add(layers.SeparableConv2D(filters=32, kernel_size=3, activation='relu',
                                 input_shape=(64, 64, 3)))
model.add(layers.SeparableConv2D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))

model.add(layers.SeparableConv2D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.SeparableConv2D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))

model.add(layers.SeparableConv2D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.SeparableConv2D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])