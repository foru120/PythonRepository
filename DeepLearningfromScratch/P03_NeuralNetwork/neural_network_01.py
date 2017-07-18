import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

X0 = np.array([1.0, 0.5])
W0 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B0 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X0, W0) + B0
Z1 = sigmoid(A1)

W1 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B1 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W1) + B1
Z2 = sigmoid(A2)

W2 = np.array([[0.1, 0.3], [0.2, 0.4]])
B2 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W2) + B2
Y = identity_function(A3)

print(Y)