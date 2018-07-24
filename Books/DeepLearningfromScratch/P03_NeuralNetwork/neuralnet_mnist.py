import sys, os
sys.path.append(os.pardir)
from DeepLearningfromScratch.Dataset.mnist import load_mnist
import numpy as np
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(a):
    return 1/(1+np.exp(-a))

def softmax(a):
    C = np.max(a)
    return np.exp(a-C)/np.sum(np.exp(a-C))

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

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # 확률이 가장 높은 원소의 인덱스를 추출
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print('Accuracy : ' + str(float(accuracy_cnt)/len(x)))