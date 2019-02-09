#todo 뉴스 기사 분류: 다중 분류 문제
#todo  ▣ 로이터 데이터셋
#todo   - 1986년에 로이터에서 공개한 짧은 뉴스 기사와 토픽의 집합 데이터 셋
#todo   - 훈련 데이터: 8,982 개 / 테스트 데이터: 2,246 개

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models

import numpy as np
import matplotlib.pyplot as plt

#todo 로이터 데이터셋 로드하기
#todo  - num_words(): 훈련 데이터에서 가장 자주 나타나는 단어 개수를 지정하는 파라미터
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#todo 로이터 데이터셋을 텍스트로 디코딩하기
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for key, value in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

#todo 데이터 인코딩하기
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train, x_test = vectorize_sequences(train_data), vectorize_sequences(test_data)
one_hot_train_labels, one_hot_test_labels = to_categorical(train_labels), to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(units=64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=46, activation='softmax'))

#todo 모델 컴파일하기
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',  # 정수 레이블 사용하는 경우: sparse_categorical_crossentropy
              metrics=['accuracy'])

#todo 검증 세트 준비하기
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#todo 모델 훈련하기
history = model.fit(x=partial_x_train,
                    y=partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))
results = model.evaluate(x=x_test,
                         y=one_hot_test_labels)
print(results)

#todo 훈련과 검증 손실 그리기
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#todo 훈련과 검증 정확도 그리기
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#todo 새로운 데이터에 대해 예측하기
predictions = model.predict(x_test)
print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))