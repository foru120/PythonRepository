print('====================================================================================================')
print('== 문제 27. 아래의 식을 파이썬으로 구현하시오.')
print('====================================================================================================\n')
import numpy as np
x = np.array([0, 1])
w = np.array([0.5, 0.5])
print(np.sum(w*x))

print('====================================================================================================')
print('== 문제 28. 위의 식에 책 52쪽에 나오는 편향을 더해서 완성한 아래의 식을 파이썬으로 구현하시오.')
print('====================================================================================================\n')
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = np.array([-0.7])
print(np.sum(w*x) + b)

print('====================================================================================================')
print('== 문제 29. and 게이트를 파이썬으로 구현하시오!')
print('====================================================================================================\n')
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    return 1 if np.sum(x*w)+b == 1 else 0

print('====================================================================================================')
print('== 문제 30. 문제 29번에 편향을 포함해서 AND 게이트 함수를 구현하시오!')
print('====================================================================================================\n')
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return 1 if np.sum(x*w)+b > 0 else 0
print('x1: 0, x2: 0 -> ', AND(0, 0))
print('x1: 0, x2: 1 -> ', AND(0, 1))
print('x1: 1, x2: 0 -> ', AND(1, 0))
print('x1: 1, x2: 1 -> ', AND(1, 1))

print('====================================================================================================')
print('== 문제 32. OR 함수를 파이썬으로 구하시오!')
print('====================================================================================================\n')
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    return 1 if np.sum(x*w)+b > 0 else 0

print('====================================================================================================')
print('== 문제 33. XOR 함수를 파이썬으로 구하시오!')
print('====================================================================================================\n')
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return 1 if np.sum(x*w)+b > 0 else 0

def XOR(x1, x2):
    return AND(OR(x1, x2), NAND(x1, x2))

print('====================================================================================================')
print('== 문제 1. NCS 평가문제')
print('====================================================================================================\n')
import numpy as np
x = np.array([1, 2])
y = np.array([3, 4])
print(2*x + y)

print('====================================================================================================')
print('== 문제 2. NCS 평가문제')
print('====================================================================================================\n')
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return 1 if np.sum(x*w)+b > 0 else 0

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return 1 if np.sum(x*w)+b > 0 else 0

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    return 1 if np.sum(x*w)+b > 0 else 0

def XOR(x1, x2):
    return AND(OR(x1, x2), NAND(x1, x2))

print('x1: 0, x2: 0 -> ', XOR(0, 0))
print('x1: 0, x2: 1 -> ', XOR(0, 1))
print('x1: 1, x2: 0 -> ', XOR(1, 0))
print('x1: 1, x2: 1 -> ', XOR(1, 1))

import numpy as np

def andPerceptron(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    netInput = x1*w1 + x2*w2
    if netInput <= theta:
        return 0
    elif netInput > theta:
        return 1

def nandPerceptron(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    netInput = x1*w1 + x2*w2
    if netInput <= theta:
        return 0
    elif netInput > theta:
        return 1

def orPerceptron(x1, x2):
    w1, w2, bias = 0.5, 0.5, -0.2
    netInput = x1*w1 + x2*w2 + bias
    if netInput <= 0:
        return 0
    else:
        return 1

def xorPerceptron(x1, x2):
    return andPerceptron(orPerceptron(x1, x2), nandPerceptron(x1, x2))

inputData = np.array([[0,0],[0,1],[1,0],[1,1]])

print("---And Perceptron---")
for xs1 in inputData:
    print(str(xs1) + " ==> " + str(andPerceptron(xs1[0], xs1[1])))

print("---Nand Perceptron---")
for xs2 in inputData:
    print(str(xs2) + " ==> " + str(nandPerceptron(xs2[0], xs2[1])))

print("---Or Perceptron---")
for xs3 in inputData:
    print(str(xs3) + " ==> " + str(orPerceptron(xs3[0], xs3[1])))

print("---XOr Perceptron---")
for xs3 in inputData:
    print(str(xs3) + " ==> " + str(xorPerceptron(xs3[0], xs3[1])))