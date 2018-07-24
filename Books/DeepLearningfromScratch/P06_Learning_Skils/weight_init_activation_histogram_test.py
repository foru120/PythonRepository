import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return np.maximum(0, x)

x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # w = np.random.randn(node_num, node_num) / np.sqrt(node_num)  # Xavier 초기값(sigmoid, tanh 활성화 함수)
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num) * np.sqrt(2)  # He 초기값(Relu 활성화 함수)

    a = np.dot(x, w)
    # z = sigmoid(a)
    z = relu_function(a)
    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + '-layer')
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()