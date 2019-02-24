#todo p.320 ~ p.322
#todo code 7-3 ~ code 7-6
#todo 7.1.3 다중 출력 모델

import numpy as np

from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(input_dim=vocabulary_size,
                                  output_dim=256)(posts_input)
x = layers.Conv1D(filters=128, kernel_size=5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(pool_size=5)(x)
x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
x = layers.MaxPooling1D(pool_size=5)(x)
x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(units=128, activation='relu')(x)

age_prediction = layers.Dense(units=1, name='age')(x)
income_prediction = layers.Dense(units=num_income_groups,
                                 activation='softmax',
                                 name='income')(x)
gender_prediction = layers.Dense(units=1,
                                 activation='sigmoid',
                                 name='gender')(x)

model = Model(inputs=posts_input, outputs=[age_prediction, income_prediction, gender_prediction])

#todo 각 출력 별 손실 함수 및 손실 가중치 설정
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])
# model.compile(optimizer='rmsprop',
#               loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'},
#               loss_weights={'age': 0.25, 'income': 1., 'gender': 10.})

#todo 다중 출력 모델에 데이터 주입하기
posts = np.array((1000, 300))
age_targets, income_targets, gender_targets = np.array((1000, 1)), np.array((1000, 10)), np.array((1000, 2))
model.fit(x=posts,
          y=[age_targets, income_targets, gender_targets],
          epochs=10,
          batch_size=64)