print('====================================================================================================')
print('== NCS 문제 1. 아래의 스크립트를 테스트해보면 x 의 원소가 out 의 원소로 변경되었을 것이다.'
      '다시 테스트할 때 x 의 원소가 out 의 원소로 변경되지 않게 하려면 어떻게 해야하는가?')
print('====================================================================================================\n')
import copy
import numpy as np

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)

out = x.copy()
print(out)
out[mask] = 0

print(out)
print(x)

# ■ 5장 목차
#    1. 역전파란 무럿인가?
#    2. 계산 그래프
#     - 덧셈 그래프
#     - 곱셈 그래프
#     - 덧셈 그래프 역전파
#     - 곱셈 그래프 역전파
#    3. 파이썬으로 단순한 계층 구현하기
#     - 덧셈 계층 (역전파 포함)
#     - 곱셈 계층 (역전파 포함)
#     - Relu 계층 (역전파 포함)
#     - sigmoid 계층 (역전파 포함)
#    4. Affine과 softmax계층 구현
#    5. 배치용 affine계층 구현
#    6. softmax wirh loss 계층
#    7. 오차역전파법 구현하기

# ■ 역전파란?
#
# 신경망 학습 처리에서 최소화되는 함수의 경사를 효율적으로 계산하기 위한 방법으로 "오류 역전파"가 있다.
#
# 함수의 경사(기울기)를 계산하는 방법
#    1. 수치 미분 <--- 너무 성능이 느림
#    2. 오류 역전파 <--- 성능이 빠르고 간단하다.
#
# * 순전파 vs 역전파
# - 순전파:  입력층  ->  은닉층 -> 출력층
# - 역전파:  출력층  ->  은닉층 -> 입력층
# 오차를 역전파시킨다.
#
# 출력층부터 차례대로 역방향으로 거슬로 올라가 각 층에 있는 노드의 오차를 계산할 수 있다.
# 각 노드의 오차를 계산하면 그 오차를 사용해서 함수의 기울기를 계산할 수 있다.
# "즉, 전파된 오차를 이용하여 가중치를 조정한다. "
#           ↓
#     오차 역전파

# ■ 계산 그래프
#
# "순전파와 역전파의 계산 과정을 그래프로 나타내는 방법"
#
# 계산 그래프의 장점?  국소적 계산을 할 수 있다.
#                             국소적 계산이란? 전체에 어떤 일이 벌어지던 상관없이 자신과 관계된
#             정보만으로 다음 결과를 출력할 수 있다는 것
#
# 그림 fig 5-4
#
# ■ 왜? 계산 그래프로 푸는가?
#    전체가 아무리 복잡해도 각 노드에서 단순한 계산에 집중하여 문제를 단순화시킬 수 있다.
#
# ■ 실제로 계산 그래프를 사용하는 가장 큰 이유는?
#    역전파를 통해서 미분을 효율적으로 계산할 수 있다.
#                   ↓
#    사과 값이 '아주 조금' 올랐을 때 '지불금액'이 얼마나 증가하는지를 알고 싶다는 것이다.
#    => 지불금액을 사과 값으로 편미분 하면 ㅇ
#               ↓
#    사과값이 1원 오르면 최종금액은 2.2원이 오른다.

print('====================================================================================================')
print('== 문제 100. 위에서 만든 곱셈 클래스를 객체화 시켜서 아래의 사과가격의 총 가격을 구하시오.')
print('====================================================================================================\n')
apple = 200
apple_num = 5
tax = 1.2

class MulLayer:
      def __init__(self):
            self.x = None
            self.y = None

      def forward(self, x, y):
            self.x = x
            self.y = y
            out = x * y
            return x * y

      def backward(self, dout):
            dx = dout * self.y
            dy = dout * self.x
            return dx, dy

apple_layer = MulLayer()
tax_layer = MulLayer()

apple_price = apple_layer.forward(apple, apple_num)
price = tax_layer.forward(apple_price, tax)
price

print('====================================================================================================')
print('== 문제 101. 덧셈 계층을 파이썬으로 구현하시오!')
print('====================================================================================================\n')
class AddLayer:
      def __init__(self):
            pass

      def forward(self, x, y):
            return x + y

      def backward(self, dout):
            dx = dout
            dy = dout
            return dx, dy

print('====================================================================================================')
print('== 문제 102. 사과 2개와 귤 5개를 구입하면 총 가격이 얼마인지 구하시오!')
print('====================================================================================================\n')
apple_node = MulLayer()
apple_price = apple_node.forward(200, 2)
orange_node = MulLayer()
orange_price = orange_node.forward(300, 5)
fruit_node = AddLayer()
fruit_price = fruit_node.forward(apple_price, orange_price)
total_node = MulLayer()
total_price = total_node.forward(fruit_price, 1.5)
print(total_price)

print('====================================================================================================')
print('== 문제 106. 문제 105번 역전파를 파이썬으로 구현하시오.')
print('====================================================================================================\n')
mul_apple_layer = MulLayer()
mul_mandarin_layer = MulLayer()
mul_pear_layer = MulLayer()
add_apple_mandarin_layer = AddLayer()
add_all_layer = AddLayer()
mul_tax_layer = MulLayer()

##순전파
apple_price = mul_apple_layer.forward(apple, apple_cnt)
mandarin_price = mul_mandarin_layer.forward(mandarin, mandarin_cnt)
pear_price = mul_pear_layer.forward(pear, pear_cnt)
apple_mandarin_price = add_apple_mandarin_layer.forward(apple_price, mandarin_price)
all_price = add_all_layer.forward(apple_mandarin_price, pear_price)
price = mul_tax_layer.forward(all_price, tax)

## 역전파
d_price = 1
d_all_price, d_tax = mul_tax_layer.backward(d_price) #6번
d_apple_mandarin_price, d_pear_price = add_all_layer.backward(d_all_price) #5번
d_apple_price, d_mandarin_price = add_apple_mandarin_layer.backward(d_apple_mandarin_price) #4번
d_apple, d_apple_cnt = mul_apple_layer.backward(d_apple_price) # 1번
d_mandarin, d_mandarin_cnt = mul_mandarin_layer.backward(d_mandarin_price) #2번
d_pear, d_pear_cnt = mul_pear_layer.backward(d_pear_price) # 3번
print(price)
print(d_apple, d_apple_cnt, d_mandarin, d_mandarin_cnt, d_pear, d_pear_cnt)

# ■ ReLU 함수를 만들기 전에 기본적으로 알아야할 문법
import copy
import numpy as np

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)

out = x.copy()
print(out)

out[mask] = 0
print(out)

print(x)

print('====================================================================================================')
print('== 문제 107. ReLU 함수를 파이썬으로 구현하시오!')
print('====================================================================================================\n')
class Relu:
      def __init__(self):
            self.mask = None

      def forward(self, x):
            self.mask = x <= 0
            out = x.copy()
            out[self.mask] = 0
            return out

      def backward(self, dout):
            dout[self.mask] = 0
            return dout

print('====================================================================================================')
print('== 문제 108. 아래의 x 변수를 생성하고 x 를 Relu 객체의 forward 함수에 넣으면 무엇이 출력되는지 확인하시오.')
print('====================================================================================================\n')
x = np.array([1.0, 5.0, -2.0, 3.0])
relu = Relu()
print(relu.forward(x))

import numpy as np
x = np.array([5, 6])
w = np.array([[2, 4, 4], [6, 3, 5]])
print(np.dot(x, w))

print('====================================================================================================')
print('== 문제 121. 문제 120번의 순전파를 구하는 함수를 forward 란 이름으로 생성하시오!')
print('====================================================================================================\n')
x = np.array([1, 2])
w = np.array([[1, 3, 5], [2, 4, 6]])
b = np.array([1, 2, 3])

def forward(x, w, b):
      return np.dot(x, w) + b

print(forward(x, w, b))

print('====================================================================================================')
print('== 문제 122. 문제 121번의 역전파를 구하는 함수를 backward 란 이름으로 생성하시오!')
print('====================================================================================================\n')
out = np.array([6, 13, 20], ndmin=2)
x = np.array([1, 2], ndmin=2)
w = np.array([[1, 3, 5], [2, 4, 6]])
b = np.array([1, 2, 3])

def backward(x, w, out):
      dx = np.dot(out, w.T)
      dw = np.dot(x.T, out)
      db = np.sum(out, axis=0)
      return dx, dw, db

print(backward(x, w, out))

print('====================================================================================================')
print('== 문제 123. 위에서 만든 forward 함수와 backward 함수를 묶어서 class 로 구성하는데 class 이름은 Affine 이라고'
      '해서 생성하시오!')
print('====================================================================================================\n')
class Affine:
      def __init__(self, w, b):
            self.w = w
            self.b = b

      def forward(self, x):
            return np.dot(x, self.w) + self.b

      def backward(self, x, out):
            dx = np.dot(out, w.T)
            dw = np.dot(x.T, out)
            db = np.sum(out)
            return dx, dw, db

a = Affine(w, b)
print(a.forward(x))
print(a.backward(x, out))

print('====================================================================================================')
print('== 문제 124. 아래의 2층 신경망의 순전파를 Affine 클래스를 사용해서 출력하시오!')
print('====================================================================================================\n')
x = np.array([1, 2], ndmin=2)
w1 = np.array([[1, 3, 5], [2, 4, 6]])
b1 = np.array([1, 2, 3])
w2 = np.array([[1, 4], [2, 5], [3, 6]])
b2 = np.array([1, 2])

a1 = Affine(w1, b1)
z1 = a1.forward(x)
a2 = Affine(w2, b2)
z2 = a2.forward(z1)
print(z2)

print('====================================================================================================')
print('== 문제 125. 아래의 2층 신경망의 역전파를 Affine 클래스를 사용해서 출력하시오!')
print('====================================================================================================\n')
x = np.array([1, 2])
w1 = np.array([[1, 3, 5], [2, 4, 6]])
b1 = np.array([1, 2, 3])
w2 = np.array([[1, 4], [2, 5], [3, 6]])
b2 = np.array([1, 2])

a1 = Affine(w1, b1)
z1 = a1.forward(x)
a2 = Affine(w2, b2)
z2 = a2.forward(z1)

dx2, dw2, db2 = a2.backward(z1, z2)
dx1, dw1, db = a1.backward(x, dx2)
print(dx1, dw1, db)

print('====================================================================================================')
print('== 문제 126. 다시 2층 신경망의 순전파를 구현하는데 은닉층에 활성화 함수로 Relu 함수를 추가해서 구현하시오!')
print('====================================================================================================\n')
x = np.array([1, 2])
w1 = np.array([[1, 3, 5], [2, 4, 6]])
b1 = np.array([1, 2, 3])
w2 = np.array([[1, 4], [2, 5], [3, 6]])
b2 = np.array([1, 2])

a1 = Affine(w1, b1)
z1 = a1.forward(x)
h1 = Relu()
z1 = h1.forward(z1)
a2 = Affine(w2, b2)
z2 = a2.forward(z1)
print(z2)

print('====================================================================================================')
print('== 문제 127. Relu 함수가 추가된 상태에서 위의 2층 신경망의 역전파를 구현하시오!')
print('====================================================================================================\n')
x = np.array([1, 2])
w1 = np.array([[1, 3, 5], [2, 4, 6]])
b1 = np.array([1, 2, 3])
w2 = np.array([[1, 4], [2, 5], [3, 6]])
b2 = np.array([1, 2])

a1 = Affine(w1, b1)
z1 = a1.forward(x)
h1 = Relu()
z1 = h1.forward(z1)
a2 = Affine(w2, b2)
z2 = a2.forward(z1)

a2.backward(h1)

print('====================================================================================================')
print('== 문제 128. 위에서 만든 softmaxWithloss 클래스를 객체화 시켜서 아래의 x (입력값), t(target value) 를 입력해서'
      '순전파 오차율을 확인하시오!')
print('====================================================================================================\n')

print('====================================================================================================')
print('== 문제 129. 데이터만 mnist 가 아니라 쉽게 하나의 값으로 변경한 코드의 순전파 결과값을 출력하시오!')
print('====================================================================================================\n')
import numpy as np
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = np.array([[1,2,3],[4,5,6]]) #(2,3)
        self.params['b1'] = np.array([1,2,3], ndmin=2) # (2, )
        self.params['W2'] = np.array([[1,2,3],[4,5,6], [7,8,9]]) #(3,3)
        self.params['b2'] = np.array([1,2,3], ndmin=2) #(2, )

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None

        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        # 순전파 ▷ X : (2, 3), W : (3, 4) -> XㆍY : (2, 4)
        # 역전파 ▷ XㆍY : (2, 4) -> X : (2, 4)ㆍWＴ, W : XＴㆍ(2, 4)
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)  # 편향은 순전파시 각각의 데이터에 더해지므로, 역전파시에 각 축의 값이 편향의 원소에 모여야 한다

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # CEE 단계로부터 back propagation 이 진행되므로, batch_size 단위로 나눠줘야한다
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

network = TwoLayerNet()
x = np.array([[1, 2], [3, 4], [5, 6]])
t = np.array([[3, 4, 5], [2, 1, 4], [2, 5, 6]])
grads = network.gradient(x, t)

print('====================================================================================================')
print('== 문제 130. 역전파된 dW 값을 출력하시오.')
print('====================================================================================================\n')
print(grads['W1'], grads['b1'], grads['W2'], grads['b2'])