#todo p.235 ~ p.241
#todo code 5-40 ~ code 5-44
#todo 5.4.3 클래스 활성화의 히트맵 시각화하기

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import keras.backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

#todo 사전 훈련된 가중치로 VGG16 네트워크 로드
model = VGG16(weights='imagenet')

#todo VGG16을 위해 입력 이미지 전처리
img_path = '/home/kyh/dataset/keras/creative_commons_elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# preds = model.predict(x)
# print('Predicted:', decode_predictions(preds, top=3))

#todo Grad-CAM 알고리즘 설정하기
african_elephant_output = model.output[:, 386]

last_conv_layer = model.get_layer('block5_conv3')

#todo block5_conv3의 특성 맵 출력에 대한 '아프리카 코끼리' 클래스의 그래디언트
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

#todo 특성 맵 채널별 그래디언트 평균값이 담긴 (512, ) 크기의 벡터
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

#todo 만들어진 특성 맵에서 채널 축을 따라 평균한 값이 클래스 활성화의 히트맵
heatmap = np.mean(conv_layer_output_value, axis=-1)

#todo 히트맵 후처리
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
# plt.show()

#todo 원본 이미지에 히트맵 덧붙이기
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'cam_dir',
                         'elephant_cam.jpg'),
            superimposed_img)