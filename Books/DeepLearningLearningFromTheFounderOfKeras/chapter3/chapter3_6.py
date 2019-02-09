#todo 주택 가격 예측: 회귀 문제
#todo  ▣ 보스턴 주택 가격 데이터 셋
#todo   - 1970년 중반 보스턴 외곽 지역의 범죄율, 지방세율 등의 데이터가 주어졌을 때 주택 가격의 중간 값을 예측하는 데이터 셋
#todo   - 훈련 데이터: 404 개 / 테스트 데이터: 102 개

from keras.datasets import boston_housing
from keras import models
from keras import layers
import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import numpy as np

#todo 보스턴 주택 가격 데이터셋
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#todo 데이터 정규화하기(Z-score) - 서로 다른 모수값(평균, 표준편차)을 가진 정규분포 집단들을 서로 비교하기 위해 정규분포를 표준화 (표준 정규분포)
#todo  - 상이한 스케일을 가진 값을 시견망에 주입하면 문제가 되므로 특성별로 정규화를 해야한다.
#todo  - 입력 데이터에 있는 각 특성에 대해서 특성의 평균을 빼고 표준 편차로 나누어서 특성의 중앙이 0 근처에 맞추어지고 표준 편차가 1이 되게 한다.
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

#todo 모델 정의하기
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # MAE(Mean Absolute Error: 평균 절대 오차)
    return model

#todo Keras session 설정
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
)

sess = tf.Session(config=config)
keras.backend.set_session(sess)

#todo K-겹 검증하기
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('처리중인 폴드 #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[: i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    model.fit(x=partial_train_data,
              y=partial_train_targets,
              epochs=num_epochs,
              batch_size=1,
              verbose=0)
    val_mse, val_mae = model.evaluate(x=val_data, y=val_targets, verbose=0)
    all_scores.append(val_mae)

print('num_epochs: ', str(num_epochs), 'all_scores:', all_scores, 'all_scores(mean): ', str(np.asarray(all_scores).mean()))

#todo 각 폴드에서 검증 점수를 로그에 저장하기
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('처리중인 폴드 #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples: ]], axis=0)
    partial_train_targets = np.concatenate([train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples: ]], axis=0)

    model = build_model()
    history = model.fit(x=partial_train_data,
                        y=partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs,
                        batch_size=1,
                        verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

#todo K-겹 검증 점수 평균을 기록하기
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#todo 처음 10개의 데이터 포인트를 제외한 검증 점수 그리기
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#todo 최종 모델 훈련하기
model = build_model()
model.fit(x=train_data,
          y=train_targets,
          epochs=80,
          batch_size=16,
          verbose=0)
test_mse_score, test_mae_score = model.evaluate(x=test_data,
                                                y=test_targets)
print(test_mae_score)

model.compile()