import sys, os
sys.path.append(os.pardir)
from DeepLearningfromScratch.dataset.mnist import load_mnist
import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

def cross_entropy_error_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 기존 1차원 배열을, 2차원 배열(1, t.size) 로 변환
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # np.arange(batch_size) : batch_size 만큼 1 간격으로 배열 생성
    # y[np.arange(batch_size), t] => y[[0, 1, 2 ..], [2, 4, 6, 0]] => one hot 인코딩에서 0 인 부분에 대해서는 계산을 하지 않았으므로,
    # 정답인 부분에 대해서만 y 값을 출력한다.
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)  # random 하게 train_size 내의 수를 batch_size 만큼 뽑아냄
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_train.shape[0], x_train.size, x_train.ndim)
print(t_train.shape)