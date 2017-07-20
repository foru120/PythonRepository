# ■ 3장 목차
#
# 1. 활성화함수(activation function)
#  - 계단 함수
#  - sigmoid 함수
#  - relu 함수
# 2. 행렬의 내적
#
# ■ 3.1 활성화 함수
# *퍼셉트론과 신경망의 차이점:
#  - 퍼셉트론: 원하는 결과를 출력하도록 가중치의 값을 적절히 정하는 작업을 사람이 수동으로 해야한다.
#  - 신경망: 가중치 매개변수의 적절한 값을 기계가 데이터로부터 자동으로 학습해서 알아낸다.
#
# 단층 퍼셉트론: 계단 함수
# 다층 퍼셉트론: sigmoid, relu....를 써야 다층의 출력: 0 or 1 의미가 생긴다.

print('====================================================================================================')
print('== 문제 35. 파이썬으로 계단함수를 구현하시오.')
print('====================================================================================================\n')
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)  # astype은 true는 1로 변경, false는 0으로 변경한다.

x_data = np.array([-1, 0, 1])
print(step_function(x_data))

print('====================================================================================================')
print('== 문제 36. 위의 step_function 함수를 이용해서 계단함수를 그리시오.')
print('====================================================================================================\n')
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)  # astype은 true는 1로 변경, false는 0으로 변경한다.

x_data = np.arange(-5, 5, 0.1)
y = step_function(x_data)

plt.plot(x_data, y)
plt.ylim(-0.1, 1.1)
plt.show()

print(step_function(x_data))

print('====================================================================================================')
print('== 문제 37. 아래와 같이 그래프가 출력되게 하시오.')
print('====================================================================================================\n')
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x < 0
    return y.astype(np.int)  # astype은 true는 1로 변경, false는 0으로 변경한다.

x_data = np.arange(-5, 5, 0.1)
y = step_function(x_data)

plt.plot(x_data, y)
plt.ylim(-0.1, 1.1)
plt.show()

print(step_function(x_data))

print('====================================================================================================')
print('== 문제 38. (점심시간 문제)')
print('====================================================================================================\n')
import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)  # astype은 true는 1로 변경, false는 0으로 변경한다.

x = np.array([-1, 0, 0])
w = np.array([0.3, 0.4, 0.1])

print(step_function(sum(x * w)))

print('====================================================================================================')
print('== 문제 39. 시그모이드 함수를 파이썬으로 구현하시오.')
print('====================================================================================================\n')
import numpy as np
def sigmoid(a):
    return 1/(1+np.exp(-a) + 0.00001)

print(sigmoid(2.0))

print('====================================================================================================')
print('== 문제 40. 시그모이드 함수를 그래프로 그리시오.')
print('====================================================================================================\n')
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(a):
    return 1/(1+np.exp(-a) + 0.00001)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

print('====================================================================================================')
print('== 문제 41. 아래와 같이 그래프가 출력되게 하시오.')
print('====================================================================================================\n')
def sigmoid(a):
    return 1/(1+np.exp(a) + 0.00001)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

print('====================================================================================================')
print('== 문제 42. 책 74쪽에 나온것처럼 계단 함수와 시그모이드 함수를 같이 출력하시오.')
print('====================================================================================================\n')
def sigmoid(a):
    return 1/(1+np.exp(-a) + 0.00001)

def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2)
plt.ylim(-0.1, 1.1)
plt.show()

print('====================================================================================================')
print('== 문제 43. Relu 함수를 생성하시오.')
print('====================================================================================================\n')
def relu(a):
    return np.maximum(a, 0)  # 둘 중에 큰 수를 출력

print(relu(-1))
print(relu(0.3))

print('====================================================================================================')
print('== 문제 44. Relu 함수를 그래프로 그리시오.')
print('====================================================================================================\n')
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.show()

print('====================================================================================================')
print('== 문제 45. 아래의 행렬 곱(행렬내적)을 파이썬으로 구현하시오.')
print('====================================================================================================\n')
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[5, 6], [7, 8], [9, 10]])
print(np.dot(a, b))

print('====================================================================================================')
print('== 문제 46. 아래의 행렬 곱(행렬내적)을 파이썬으로 구현하시오.')
print('====================================================================================================\n')
a = np.array([[5, 6], [7, 8], [9, 10]])
b = np.array([[1], [2]])
print(np.dot(a, b))

print('====================================================================================================')
print('== 문제 47. 아래의 그림을 numpy 로 구현하시오.')
print('====================================================================================================\n')
x = np.array([1, 2])
w = np.array([[1, 3, 5], [2, 4, 6]])
b = np.array([7, 8, 9])
print(np.dot(x, w) + b)

print('====================================================================================================')
print('== 문제 48. 위의 문제에서 구한 입력신호의 가중의 합인 y 값이 활성함수인 sigmoid 함수를 통과하면 어떤값으로 ')
print('== 출력되는지 z 값을 확인하시오. ')
print('====================================================================================================\n')
x = np.array([1, 2])
w = np.array([[1, 3, 5], [2, 4, 6]])
b = np.array([7, 8, 9])
y = np.dot(x, w) + b
print(sigmoid(y))

print('====================================================================================================')
print('== 문제 49. 아래의 신경망 그림을 파이썬으로 구현하시오.')
print('====================================================================================================\n')
x = np.array([4.5, 6.2])
w1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.array([0.7, 0.8])
w2 = np.array([[0.5, 0.6], [0.7, 0.8]])
b2 = np.array([0.7, 0.8])
w3 = np.array([[0.1, 0.2], [0.3, 0.4]])
b3 = np.array([0.7, 0.8])

a1 = np.dot(x, w1) + b1
L1 = sigmoid(a1)
a2 = np.dot(L1, w2) + b2
L2 = sigmoid(a2)
output = np.dot(L2, w3) + b3
print(output)

print('====================================================================================================')
print('== 문제 50. softmax 함수를 파이썬으로 구현하시오.')
print('====================================================================================================\n')
def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)

print('====================================================================================================')
print('== 문제 52. 책 88 페이지에 나오는 3층 신경망을 파이썬으로 구현하시오!')
print('====================================================================================================\n')
def hidden_layer(x, w, b):
    a = np.dot(x, w) + b
    return sigmoid(a)

x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])
w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

L1 = hidden_layer(x, w1, b1)
L2 = hidden_layer(L1, w2, b2)
y_ = np.dot(L2, w3) + b3
print(y_)

print('====================================================================================================')
print('== NCS 문제 1. 아래의 행렬곱을 파이썬으로 구현하는데 두 가지 방법으로 구현하시오.')
print('====================================================================================================\n')
import numpy as np
A = np.array([[1, 1, -1], [4, 0, 2], [1, 0, 0]])
B = np.array([[2, -1], [3, -2], [0, 1]])
A1 = np.matrix([[1, 1, -1], [4, 0, 2], [1, 0, 0]])
B1 = np.matrix([[2, -1], [3, -2], [0, 1]])
print(np.dot(A, B))
print(A1 * B1)

print('====================================================================================================')
print('== NCS 문제 2. 아래의 신경망을 파이썬으로 구현하시오!')
print('====================================================================================================\n')
x = np.array([0.2, 0.7, 0.9])
w = np.array([[2, 4, 3], [2, 3, 5], [2, 4, 4]])
b = np.array([-3, 4, 9])

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))

y = softmax(np.dot(x, w) + b)
print(y)

print('====================================================================================================')
print('== 문제 53. x_train 의 0 번째 요소의 필기체 숫자는 5 였다. 그렇다면 x_train 의 1번째 요소의 필기체 숫자는 무엇인지 확인하시오.')
print('====================================================================================================\n')
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[5]
label = t_train[5]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)

print('====================================================================================================')
print('== 문제 54. 훈련 이미지가 60000 장이 맞는지 확인해보시오.')
print('====================================================================================================\n')
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print(len(x_train))

print('====================================================================================================')
print('== 문제 55. (점심시간 문제) 필기체 숫자 9를 출력하시오.')
print('====================================================================================================\n')
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from DeepLearningClass.dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

idx = list(t_train).index(9)
img = x_train[idx]
label = t_train[idx]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)

print('====================================================================================================')
print('== 문제 56. 위의 코드를 수정해서 하나의 x[34] 의 테스트 데이터가 신경망이 예측한 것과 맞는지 확인하시오.')
print('====================================================================================================\n')
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0

y = predict(network, x[34])
print(np.argmax(y), t[34])

print('====================================================================================================')
print('== 문제 57. 아래와 같이 결과를 출력하시오.')
print('====================================================================================================\n')
print([v for v in range(0, 10, 3)])

print('====================================================================================================')
print('== 문제 58. 아래의 리스트를 만들고 이중에 최대값의 원소의 인덱스를 출력하시오.')
print('====================================================================================================\n')
a = [v for v in range(0, 10, 3)]
print(np.argmax(a))

print('====================================================================================================')
print('== 문제 60. 아래의 행렬 배열을 생성하고 각 행의 최대값에 해당하는 인덱스가 출력되게 하시오.')
print('====================================================================================================\n')
a = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
print(np.argmax(a, axis=1))

print('====================================================================================================')
print('== 문제 61. 아래의 두개의 리스트를 만들고 서로 같은 자리에 같은 숫자가 몇개가 있는지 출력하시오.')
print('====================================================================================================\n')
a = np.array([2, 1, 3, 5, 1, 4, 2, 1, 1, 0])
b = np.array([2, 1, 3, 4, 5, 4, 2, 1, 1, 2])
print(np.sum(a == b))

print('====================================================================================================')
print('== 문제 62. 아래의 리스트를 x 라는 변수에 담고 앞에 5개의 숫자만 출력하시오.')
print('====================================================================================================\n')
a = [1,2,3,4,5,6,7,8,9,10]
a[:5]

print('====================================================================================================')
print('== 문제 63. 100장의 이미지를 한번에 입력층에 넣어서 추론하는 신경망 코드를 수행하시오.')
print('====================================================================================================\n')
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
import pickle
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100  # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

print('====================================================================================================')
print('== 문제 64. 100장의 이미지를 한번에 입력층에 넣어서 추론하는 신경망 코드를 수행하시오.')
print('====================================================================================================\n')
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록
import numpy as np
import pickle
from DeepLearningClass.dataset.mnist import load_mnist
from DeepLearningClass.common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100  # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

print('====================================================================================================')
print('== 문제 65. 훈련 데이터로 (6만개)로 batch_size 1 로 했을때와 batch_size 100 으로 했을때의 정확도와 수행속도의 차이를 비교하시오.')
print('====================================================================================================\n')
