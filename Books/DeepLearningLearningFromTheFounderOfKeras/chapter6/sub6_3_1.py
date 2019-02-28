#todo p.279 ~ p.282
#todo code 6-28 ~ code 6-31
#todo 6.3.1 기온 예측 문제

import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'G:/04.dataset/09.jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.show()