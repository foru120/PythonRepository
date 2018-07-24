#todo 순차형 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

model = Sequential()

model.add(Dense(units=64, input_dim=784))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model.fit(x=x_train, y=y_train, epochs=10, batch_size=64, callbacks=[TensorBoard(log_dir='./MNIST_keras_board/basic'), early_stop])

loss_and_metrics = model.evaluate(x=x_test, y=y_test, batch_size=64)
classes = model.predict(x=x_test, batch_size=64)

#todo 함수형 모델
'''
    순차형 모델과의 주요 차이점은, 함수형 모델에서는 먼저 입력과 출력을 정의하고 난 다음 모델을 인스턴스화 한다.
'''
inputs = Input(shape=(784, ))

x = Dense(units=64, activation='relu')(inputs)
x = Dense(units=32, activation='relu')(x)
outputs = Dense(units=10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10, batch_size=64)
loss_and_metrics = model.evaluate(x=x_test, y=y_test, batch_size=64)
classes = model.predict(x=x_test, batch_size=64)