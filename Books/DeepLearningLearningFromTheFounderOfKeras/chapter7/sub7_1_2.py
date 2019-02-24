#todo p.317 ~ p.319
#todo code 7-1 ~ code 7-2
#todo 7.1.2 다중 입력 모델

import numpy as np

from keras.utils import to_categorical
from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

#todo Text Encoding
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(input_dim=text_vocabulary_size,
                                 output_dim=64)(text_input)
encoded_text = layers.LSTM(units=32)(embedded_text)

#todo Question Encoding
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(input_dim=question_vocabulary_size,
                                     output_dim=32)(question_input)
encoded_question = layers.LSTM(units=16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

answer = layers.Dense(units=answer_vocabulary_size,
                      activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

num_samples = 1000
max_length = 100

text = np.random.randint(low=1, high=text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(low=1, high=question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(low=0, high=answer_vocabulary_size, size=num_samples)
answers = to_categorical(y=answers, num_classes=answer_vocabulary_size)

model.fit(x=[text, question], y=answers, epochs=10, batch_size=128)
# model.fit(x={'text': text, 'question': question}, y=answers, epochs=10, batch_size=128)