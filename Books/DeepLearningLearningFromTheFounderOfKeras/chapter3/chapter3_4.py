#todo 영화 리뷰 분류: 이진 분류 예제
#todo  ▣ IMDB 데이터 셋
#todo   - 인터넷 영화 데이터베이스로부터 가져온 양극단의 리뷰 5만 개로 이루어진 데이터 셋
#todo   - 훈련 데이터: 25,000 개 / 테스트 데이터 25,000 개 (부정 50%, 긍정 50%)

from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
import numpy as np

#todo IMDB 데이터 셋을 훈련 데이터와 테스트 데이터로 분할
#todo  - num_words(): 훈련 데이터에서 가장 자주 나타나는 단어 개수를 지정하는 파라미터
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#todo train_data 에 포함된 인코딩 된 단어 시퀀스를 원래 단어로 복원
#todo  - get_word_index(): 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

#todo 정수 시퀀스를 이진 행렬로 인코딩하기
#todo  - [2, 4, 6 ...] -> [0, 1, 0, 1, 0, 1 ...] 로 Embedding
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train, y_train = vectorize_sequences(train_data), np.asarray(train_labels).astype(np.float32)
x_test, y_test = vectorize_sequences(test_data), np.asarray(test_labels).astype(np.float32)

#todo 모델 정의하기
model = models.Sequential()
model.add(layers.Dense(units=16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

#todo 모델 컴파일하기
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#todo 옵티마이저 설정하기
# model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

#todo 손실과 측정을 사용자 함수로 지정하기
#todo  - loss 와 metrics 매개변수에 함수 객체를 전달
# model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

#todo 검증 세트 준비하기
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#todo 모델 훈련하기
history = model.fit(x=partial_x_train,
                    y=partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#todo 모델 테스트
results = model.evaluate(x_test, y_test)
print(results)

#todo 훈련과 검증 손실 그리기
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#todo 훈련과 검증 정확도 그리기
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#todo 훈련된 모델로 새로운 데이터에 대해 예측하기
model.predict(x_test)