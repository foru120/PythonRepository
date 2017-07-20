# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 1. flatten
#  - true : 입력 이미지를 평탄하게 1차원 배열로 변환하는 것
#  - false : 입력 이미지를 평탄하게 1차원 배열로 변환하지 않는 것
# 2. normalize
#  - true : 픽셀의 값을 0 ~ 1 사이로 변환
#  - false : 픽셀의 값을 0 ~ 255 사이로 그대로 둔다
# 3. one_hot_label
#  - one-hot encoding 형태로 저장할지를 정함
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)