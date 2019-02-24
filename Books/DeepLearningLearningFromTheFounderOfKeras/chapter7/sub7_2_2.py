#-*-coding: utf-8-*-
#todo p.333 ~ p.340
#todo code 7-7 ~ code 7-9
#todo 7.2.2 텐서보드 소개: 텐서플로의 시각화 프레임워크

import os
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 2000  # 특성으로 사용할 단어의 수
max_len = 500  # 사용할 텍스트의 길이

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(sequences=x_train,
                                 maxlen=max_len)
x_test = sequence.pad_sequences(sequences=x_test,
                                maxlen=max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(input_dim=max_features,
                           output_dim=128,
                           input_length=max_len,
                           name='embed'))
model.add(layers.Conv1D(filters=32, kernel_size=7, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=5))
model.add(layers.Conv1D(filters=32, kernel_size=7, activation='relu'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# ▣ keras.utils.plot_model
# - 모델의 층 그래프를 그려 주는 기능
from keras.utils import plot_model
plot_model(model=model,
           to_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_graph', 'model_graph.png'),
           show_shapes=True)

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tensorboard_log'),
        histogram_freq=1,  # 1 에포크마다 활성화 출력의 히스토그램을 기록
        embeddings_freq=1  # 1 에포크마다 임베딩 데이터를 기록
    )
]

history = model.fit(
    x=x_train,
    y=y_train,
    epochs=1,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks
)