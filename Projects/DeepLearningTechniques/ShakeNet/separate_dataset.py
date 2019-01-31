import os
import numpy as np
import re
import shutil
from collections import defaultdict

asis_root_path = 'G:/04_dataset/cifar10'
tobe_root_path = 'G:/04_dataset/cifar10_tt'

#todo cifar10 dataset train/test/validation => (클래스 당 4500/1000/500)
train_file_paths = defaultdict(list)
for (path, dirs, files) in os.walk(asis_root_path):
    for file in files:
        train_file_paths[re.match('.*\\\(\d+)', path).group(1)].append(os.path.join(path, file))

tot_data_len = len(train_file_paths.get(list(train_file_paths.keys())[0]))
train_len = 4500
test_len = 1000
valid_len = 500

#todo 무작위로 train/test/validation 데이터 분할
tot_random_sort = np.random.permutation(tot_data_len)
train_random_sort = tot_random_sort[: train_len]
test_random_sort = tot_random_sort[train_len: train_len+test_len]
valid_random_sort = tot_random_sort[train_len+test_len: ]

train_x, train_y = [], []
test_x, test_y = [], []
valid_x, valid_y = [], []

for key in train_file_paths.keys():
    train_file_path = np.asarray(train_file_paths.get(key))
    train_x.append(list(train_file_path[train_random_sort]))
    train_y.append([key] * train_len)
    test_x.append(list(train_file_path[test_random_sort]))
    test_y.append([key] * test_len)
    valid_x.append(list(train_file_path[valid_random_sort]))
    valid_y.append([key] * valid_len)

train_x, train_y = np.asarray(train_x).flatten(), np.asarray(train_y, dtype=np.int64).flatten()
test_x, test_y = np.asarray(test_x).flatten(), np.asarray(test_y, dtype=np.int64).flatten()
valid_x, valid_y = np.asarray(valid_x).flatten(), np.asarray(valid_y, dtype=np.int64).flatten()

for x, y in zip(train_x, train_y):
    os.makedirs(os.path.join(tobe_root_path, 'train', str(y)), exist_ok=True)
    shutil.copy(x, os.path.join(tobe_root_path, 'train', str(y)))

for x, y in zip(test_x, test_y):
    os.makedirs(os.path.join(tobe_root_path, 'test', str(y)), exist_ok=True)
    shutil.copy(x, os.path.join(tobe_root_path, 'test', str(y)))

for x, y in zip(valid_x, valid_y):
    os.makedirs(os.path.join(tobe_root_path, 'validation', str(y)), exist_ok=True)
    shutil.copy(x, os.path.join(tobe_root_path, 'validation', str(y)))