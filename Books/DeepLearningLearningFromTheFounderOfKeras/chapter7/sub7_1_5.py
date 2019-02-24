#todo p.327 ~ p.327
#todo code x ~ code x
#todo 7.1.5 층 가중치 공유

from keras import layers
from keras import Input
from keras.models import Model

lstm = layers.LSTM(units=32)
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)  # 기존 층 객체를 호출하면 가중치가 재사용 됨

merged = layers.concatenate(input=[left_output, right_output], axis=-1)
predictions = layers.Dense(units=1, activation='sigmoid')(merged)

model = Model(inputs=[left_input, right_input], outputs=predictions)