#todo p.282 ~ p.299
#todo code 6-32 ~ code 6-44
#todo 6.3.2 데이터 준비

import os
import numpy as np

data_dir = 'G:/04.dataset/09.jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

mean = float_data[:200000].mean(axis=0)
std = float_data[:200000].std(axis=0)
float_data -= mean
float_data /= std

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    """
    :param data: 원본 데이터 배열
    :param lookback: 입력으로 사용하기 위해 거슬러 올라갈 타임스텝
    :param delay: 타깃으로 사용할 미래의 타임스텝
    :param min_index: 추출할 타임스텝의 범위를 지정하기 위한 data 배열의 인덱스. 검증 데이터와 테스트 데이터를 분리하는 데 사용
    :param max_index: 추출할 타임스텝의 범위를 지정하기 위한 data 배열의 인덱스. 검증 데이터와 테스트 데이터를 분리하는 데 사용
    :param shuffle: 샘플을 섞을지, 시간 순서대로 추출할지를 결정
    :param batch_size: 배치의 샘플 수
    :param step: 데이터를 샘플링할 타임스텝 간격. 1시간에 하나의 데이터 포인트를 추출하기 위해 기본값 6으로 지정
    :return:
    """
    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + lookback

    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(data=float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(data=float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(data=float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

#todo 상식 수준의 기준점 (loss: 0.29)
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print('상식 수준의 기준점:', np.mean(batch_maes) * std[1])

# evaluate_naive_method()

#todo 기본적인 머신 러닝 방법 (loss: 0.3378)
# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(generator=train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

#todo 첫 번째 순환 신경망-GRU (loss: 0.2980)
# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(generator=train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

#todo 드롭아웃 규제된 GPU를 사용한 모델을 훈련하고 평가하기 (loss: 0.2702)
# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.GRU(32,
#                      dropout=0.2,
#                      recurrent_dropout=0.2,
#                      input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(generator=train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

#todo 스태킹 순환 층 (loss: 0.2686)
# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.GRU(32,
#                      dropout=0.1,
#                      recurrent_dropout=0.5,
#                      return_sequences=True,
#                      input_shape=(None, float_data.shape[-1])))
# model.add(layers.GRU(64,
#                      activation='relu',
#                      dropout=0.1,
#                      recurrent_dropout=0.5))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(generator=train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

#todo 양방향 LSTM 을 훈련하고 평가하기 (loss: )
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()