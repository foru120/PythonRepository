import tensorflow as tf
import os
import numpy as np
from scipy import misc
from collections import defaultdict

class DataLoader:
    def __init__(self, data_root_path, batch_size=50, n_way=5, k_shot=1, train_mode=True):
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot

        img_data = []
        img_dict = defaultdict(list)

        '''Omniglot 파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(data_root_path):
            for file in files:
                img_dict[path].append(os.path.join(path, file))

        for img_paths in img_dict.values():
            img_data.append(self.aug_rotate_img(img_paths))
        img_dict.clear()

        img_data = np.expand_dims(np.asarray(img_data), -1)
        np.random.shuffle(img_data)

        if train_mode:
            self.x = img_data[:1200, :, :, :, :]
            self.num_classes = self.x.shape[0]
            self.num_samples = self.x.shape[1]
        else:
            self.x = img_data[1200:, :, :, :, :]
            self.num_classes = self.x.shape[0]
            self.num_samples = self.x.shape[1]

        self.iters = self.num_classes

    def aug_rotate_img(self, img_paths):
        data = []
        for img_path in img_paths:
            img = misc.imresize(misc.imread(name=img_path, mode='L'), (28, 28))
            for angle in (0, 90, 180, 270):
                data.append(misc.imrotate(arr=img, angle=angle, interp='nearest') / 255.)
        return data

    def next_batch(self):
        x_set_batch, y_set_batch = [], []
        x_hat_batch, y_hat_batch = [], []
        for _ in range(self.batch_size):
            x_set, y_set = [], []
            classes = np.random.permutation(self.num_classes)[:self.n_way]
            target_class = np.random.randint(self.n_way)
            for i, c in enumerate(classes):
                samples = np.random.permutation(self.num_samples)[:self.k_shot+1]
                for sample in samples[:-1]:
                    x_set.append(self.x[c][sample]), y_set.append(i)
                if i == target_class:
                    x_hat_batch.append(self.x[c][samples[-1]])
                    y_hat_batch.append(i)
            x_set_batch.append(x_set)
            y_set_batch.append(y_set)
        return np.asarray(x_set_batch).astype(np.float32), np.asarray(y_set_batch).astype(np.int32), \
               np.asarray(x_hat_batch).astype(np.float32), np.asarray(y_hat_batch).astype(np.int32)

# loader = DataLoader(data_root_path='D:\\100_dataset\\omniglot', batch_size=50, n_way=20, k_shot=1, train_mode=True)
# support_set_x, support_set_y, example_set_x, example_set_y = loader.next_batch()
# print(support_set_x.shape, support_set_y.shape, example_set_x.shape, example_set_y.shape)