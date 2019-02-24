#todo p.229 ~ p.235
#todo code 5-32 ~ code 5-39
#todo 5.4.2 컨브넷 필터 시각화하기

import numpy as np
import matplotlib.pyplot as plt

from keras.applications import VGG16
from keras import backend as K

#todo 모델 정의
model = VGG16(include_top=False,
              weights='imagenet')

#todo 텐서를 이미지 형태로 변환하기 위한 유틸리티 함수
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#todo 필터 시각화 이미지를 만드는 함수
def generate_pattern(layer_name, filter_index, size=150):
    #todo 주어진 층과 필터의 활성화를 최대화하기 위한 손실 함수를 정의
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    #todo 손실에 대한 입력 이미지의 그래디언트를 계산
    grads = K.gradients(loss, model.input)[0]

    #todo 그래디언트 정규화
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    #todo 입력 이미지에 대한 손실과 그래디언트를 반환
    iterate = K.function([model.input], [loss, grads])

    #todo 잡음이 섞인 회색 이미지로 시작
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    #todo 경사 상승법 수행
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

#todo 층에 있는 각 필터에 반응하는 패턴 생성하기
layer_name = 'block5_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3), dtype='uint8')

for i in range(8):
    for j in range(8):
        #todo layer_name에 있는 i + (j*8)번째 필터에 대한 패턴을 생성
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()