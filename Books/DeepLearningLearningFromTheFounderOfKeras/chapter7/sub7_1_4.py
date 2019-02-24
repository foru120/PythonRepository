#todo p.322 ~ p.326
#todo code x ~ code x
#todo 7.1.4 층으로 구성된 비순환 유향 그래프

# ▣ 인셉션 모듈
# - 네트워크가 따로따로 공간 특성과 채널 방향의 특성을 학습
# - 1x1 합성곱으로 시작해서 3x3 합성곱이 뒤따르고 마지막에 전체 출력 특성이 합쳐지는 구조
# - 채널이 공간 방향으로 상관관계가 크고 채널 간에는 독립적이라고 가장하면 좋은 전략 구조

# ※ 1x1 합성곱의 목적
# - 입력 텐서의 채널 정보를 혼합한 특성을 계산하며, 공간 방향으로는 정보를 섞지 않음

#todo 인셉션 모듈 구현
from keras import layers, Input

x = Input(shape=(224, 224, 3), dtype='float32', name='x')
branch_a = layers.Conv2D(filters=128, kernel_size=1, activation='relu', strides=2)(x)

branch_b = layers.Conv2D(filters=128, kernel_size=1, activation='relu')(x)
branch_b = layers.Conv2D(filters=128, kernel_size=3, activation='relu', strides=2)(branch_b)

branch_c = layers.AveragePooling2D(pool_size=3, strides=2)(x)
branch_c = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(branch_c)

branch_d = layers.Conv2D(filters=128, kernel_size=1, activation='relu')(x)
branch_d = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(branch_d)
branch_d = layers.Conv2D(filters=128, kernel_size=3, activation='relu', strides=2)(branch_d)

output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

# ▣ 잔차 모듈
# - 하위 층을 출력을 상위 층의 입력으로 사용
# - summation 연산으로 하위 층과 상위 층의 출력이 동일해야 하며, 크기가 다르면 선형 변환을 사용하여 하위 층의 활성화 출력을
#   목표 크기로 변환

#todo 잔차 모듈 구현
from keras import layers, Input

x = Input(shape=(224, 224, 3), dtype='float32', name='x')
y = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
y = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(y)
y = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(y)

y = layers.add([y, x])

# ※ 딥러닝의 표현 병목
# - Sequential 모델에서 특정 층은 이전 층의 활성화 출력 정보만 사용하므로, 이전 층의 출력이 너무 작으면 정보가 적어 성능이
#   떨어질 수 있음
#   위에서 사용한 잔차 모듈로 하위 층의 정보를 다시 주입하면 해당 표현 병목 현상을 어느 정도 해결 가능
