import tensorflow as tf
import os
import numpy as np
import re

class DataLoader:

    def __init__(self, batch_size, train_root_path, test_root_path):
        self.batch_size = batch_size

        train_files, test_files = [], []

        for (path, dirs, files) in os.walk(train_root_path):
            for file in files:
                train_files.append(os.path.join(path, file))

        for (path, dirs, files) in os.walk(test_root_path):
            for file in files:
                test_files.append(os.path.join(path, file))

        self.train_x_len = len(train_files)
        self.test_x_len = len(test_files)

        train_x = np.array(train_files)[np.random.permutation(self.train_x_len)]
        train_y = [re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1) for x in train_x]

        test_x = np.array(test_files)[np.random.permutation(self.test_x_len)]
        test_y = [re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1) for x in test_x]

        with tf.variable_scope('dataloader'):
            self.train_x = tf.convert_to_tensor(train_x, dtype=tf.string, name='train_x')
            self.train_y = tf.convert_to_tensor(train_y, dtype=tf.int64, name='train_y')

            self.test_x = tf.convert_to_tensor(test_x, dtype=tf.string, name='test_x')
            self.test_y = tf.convert_to_tensor(test_y, dtype=tf.string, name='test_y')

    def train_setter(self, x, y):
        with tf.variable_scope('train_setter'):
            # img = tf.cast(tf.image.resize_images(tf.image.decode_png(tf.read_file(x_path), channels=3, name='image'), size=(192, 256)), tf.float32)
            img = tf.cast(tf.image.resize_images(tf.transpose(tf.image.decode_png(tf.read_file(x), channels=3, name='image'), perm=[1, 0, 2]), size=(180, 60)), tf.float32)
            scaled_img = tf.subtract(tf.divide(img, 127.5), 1)

        return scaled_img, y

    def train_loader(self):
        with tf.variable_scope('train_loader'):
            # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).repeat()

            # 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다
            dataset_map = dataset.map(self.train_setter).batch(self.batch_size)

            # 데이터셋을 이터레이터를 통해 지속적으로 불러준다
            iterator = dataset_map.make_one_shot_iterator()

            # 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다.
            batch_input = iterator.get_next()

        return batch_input

    def test_setter(self, x, y):
        with tf.variable_scope('test_setter'):
            # img = tf.cast(tf.image.resize_images(tf.image.decode_png(tf.read_file(x_path), channels=3, name='image'), size=(192, 256)), tf.float32)
            img = tf.cast(tf.image.resize_images(tf.transpose(tf.image.decode_png(tf.read_file(x), channels=3, name='image'), perm=[1, 0, 2]), size=(180, 60)), tf.float32)
            scaled_img = tf.subtract(tf.divide(img, 127.5), 1)

        return scaled_img, y

    def test_loader(self):
        with tf.variable_scope('test_loader'):
            # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).repeat()

            # 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다
            dataset_map = dataset.map(self.train_setter).batch(self.batch_size)

            # 데이터셋을 이터레이터를 통해 지속적으로 불러준다
            iterator = dataset_map.make_one_shot_iterator()

            # 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다.
            batch_input = iterator.get_next()

        return batch_input