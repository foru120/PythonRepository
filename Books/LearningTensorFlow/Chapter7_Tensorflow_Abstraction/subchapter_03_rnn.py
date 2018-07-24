import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

#todo IMDb 데이터셋 로딩
train, test, _ = imdb.load_data(path='./imdb_data/imdb.pkl', n_words=10000, valid_portion=0.1)
X_train, Y_train = train
X_test, Y_test = test

#todo 시퀀스 제로 패딩 및 레이블 이진 벡터 변환
X_train = pad_sequences(sequences=X_train, maxlen=100, value=0.)
X_test = pad_sequences(sequences=X_test, maxlen=100, value=0.)
Y_train = to_categorical(y=Y_train, nb_classes=2)
Y_test = to_categorical(y=Y_test, nb_classes=2)

#todo LSTM 네트워크 구성
RNN = tflearn.input_data([None, 100])
RNN = tflearn.embedding(incoming=RNN, input_dim=10000, output_dim=128)

RNN = tflearn.lstm(incoming=RNN, n_units=128, dropout=0.8)
RNN = tflearn.fully_connected(incoming=RNN, n_units=2, activation='softmax')
RNN = tflearn.regression(incoming=RNN, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

#todo 네트워크 학습
model = tflearn.DNN(network=RNN, tensorboard_verbose=0,
                    tensorboard_dir='./MNIST_tflearn_board/rnn',
                    checkpoint_path='./MNIST_tflearn_checkpoints/rnn')
model.fit(X_train, Y_train, validation_set=(X_test, Y_test), show_metric=True, batch_size=32)