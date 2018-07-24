import numpy as np

def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a-C)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a

a = np.array([1010, 1000, 990])

print(softmax(a))