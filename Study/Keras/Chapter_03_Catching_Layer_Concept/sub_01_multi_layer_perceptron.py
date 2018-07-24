import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 랜덤시트 고정시키기
np.random.seed(5)

dataset = np.loadtxt('./dataset/diabetes.csv', delimiter=',')