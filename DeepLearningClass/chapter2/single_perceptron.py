import numpy as np

class SinglePerceptron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def predict(self, x):
        a = np.dot(x, self.w) + self.b
        return self.step_function(a)

    def step_function(self, a):
        return np.array(a > 0, dtype=np.int)

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [0], [0], [1]]
w = np.array([[0.4], [0.1]])
b = np.array([-0.3]).reshape((1, 1))
learning_rate = 0.05
layer = SinglePerceptron(w, b)

while True:
    cost = 0
    for idx in range(len(x_data)):
        y_ = layer.predict(np.array(x_data[idx]).reshape((1, 2)))
        cost += y_data[idx] - y_
        layer.w += learning_rate * x_data[idx] * (y_data[idx] - y_)
        layer.b -= learning_rate * (y_data[idx] - y_)
        print(layer.w, layer.b, cost)

    if cost == 0:
        break
