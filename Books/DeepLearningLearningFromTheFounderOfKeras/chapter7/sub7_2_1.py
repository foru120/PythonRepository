#todo p.330 ~ p.333
#todo code x ~ code x
#todo 7.2.1 콜백을 사용하여 모델의 훈련 과정 제어하기

# ▣ ModelCheckPoint 와 EarlyStopping 콜백
# - ModelCheckPoint: 훈련하는 동안 모델을 계속 저장하는 콜백 함수
# - EarlyStopping: 정해진 에포크 동안 모니터링 지표가 향상되지 않을 때 훈련을 중지하는 콜백 함수

import os
import keras
from keras.models import Model

x, y = [], []
x_val, y_val = [], []

callback_list = [
    keras.callbacks.EarlyStopping(  # 성능 향상이 멈추면 훈련을 중지
        monitor='val_acc',  # 모델의 검증 정확도를 모니터링
        patience=1  # 1 에포크보다 더 길게(즉 2 에포크 동안) 정확도가 향상되지 않으면 훈련이 중지
    ),
    keras.callbacks.ModelCheckpoint(  # 에포크마다 현재 가중치를 저장
        filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_log', 'model.h5'),  # 모델 파일의 경로
        monitor='val_loss',  # 이 두 매개변수는 val_loss 가 좋아지지 않으면 모델 파일을 덮어쓰지 않는다는 뜻.
        save_best_only=True  # 훈련하는 동안 가장 좋은 모델이 저장.
    )
]

model = Model()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit(
    x=x,
    y=y,
    epochs=10,
    batch_size=32,
    callbacks=callback_list,
    validation_data=(x_val, y_val)
)

# ▣ ReduceLROnPlateau 콜백
# - 검증 손실이 향상되지 않을 때 학습률을 작게 할 수 있는 콜백 함수

callback_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # 모델의 검증 손실을 모니터링
        factor=0.1,  # 콜백이 홀출될 때 학습률을 10배로 줄임
        patience=10  # 검증 손실이 10 에포크 동안 좋아지지 않으면 콜백이 호출
    )
]

model.fit(
    x=x,
    y=y,
    epochs=10,
    batch_size=32,
    callbacks=callback_list,
    validation_data=(x_val, y_val)
)

# ▣ 사용자 콜백 함수
# - 내장 콜백에서 제공하지 않는 특수한 행동이 훈련 도중 필요하면 자신만의 콜백을 만들 수 있음
# - 콜백은 keras.callbacks.Callback 클래스를 상속받아 구현
# - 훈련시 호출 지점
#  - on_epoch_begin: 각 에포크가 시작할 때 호출
#  - on_epoch_end: 각 에포크가 끝날 때 호출
#  - on_batch_begin: 각 배치 처리가 시작되기 전에 호출
#  - on_batch_end: 각 배치 처리가 끝난 후에 호출
#  - on_train_begin: 훈련이 시작될 때 호출
#  - on_train_end: 훈련이 끝날 때 호출

import numpy as np

class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(inputs=model.input,
                                                    outputs=layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')

        validation_sample = self.validation_data[0][0:1]  # validation_data 의 첫 번째 원소는 입력 데이터고, 두 번째 원소는 레이블
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'wb')
        np.savez(f, activations)
        f.close()
