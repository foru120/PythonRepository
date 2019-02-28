#todo p.266 ~ p.271
#todo code 6-22 ~ code 6-24
#todo 6.2.1 케라스의 순환 층

import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, SimpleRNN
from keras.models import Sequential

max_features = 10000
maxlen = 500
batch_size = 32

print('데이터 로딩...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), '훈련 시퀀스')
print(len(x_test), '테스트 시퀀스')

print('시퀀스 패딩 (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)  # maxlen 에 설정된 길이보다 길면 자르고, 짧으면 0으로 패딩
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train 크기:', x_train.shape)
print('x_test 크기:', x_test.shape)

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x=x_train, y=y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()