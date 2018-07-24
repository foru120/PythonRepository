import numpy as np

def predict(x, w):
    a = np.sum(x * w)
    return step_function(a)

def step_function(a):
    return np.array(a >= 0, dtype=np.int)

x_data = np.array([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]])
y_data = np.array([0, 0, 0, 1])
w = np.array([0.3, 0.4, 0.1])

learning_rate = 0.05

while True:
    epoch_cost = 0
    for idx, x in enumerate(x_data):
        y_ = predict(x, w)
        cost = y_data[idx] - y_
        w = w + learning_rate * x * cost
        epoch_cost += cost

    if epoch_cost == 0:
        break
print(w)