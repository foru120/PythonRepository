import tensorflow as tf
import os
import numpy as np
import re

class DataLoader:
    def __init__(self, batch_size, train_right_root_path, test_right_root_path, train_left_root_path, test_left_root_path):
        self.batch_size = batch_size

        train_right_files, test_right_files, train_left_files, test_left_files = [], [], [], []

        '''오른쪽/왼쪽 눈 파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(train_right_root_path):
            for file in files:
                train_right_files.append(os.path.join(path, file))
        for (path, dirs, files) in os.walk(test_right_root_path):
            for file in files:
                test_right_files.append(os.path.join(path, file))

        for (path, dirs, files) in os.walk(train_left_root_path):
            for file in files:
                train_left_files.append(os.path.join(path, file))
        for (path, dirs, files) in os.walk(test_left_root_path):
            for file in files:
                test_left_files.append(os.path.join(path, file))

        '''오른쪽/왼쪽 눈 파일 개수'''
        self.train_right_x_len = len(train_right_files)
        self.test_right_x_len = len(test_right_files)

        self.train_left_x_len = len(train_left_files)
        self.test_left_x_len = len(test_left_files)

        '''오른쪽/왼쪽 눈 x, y 데이터 설정'''
        train_random_sort = np.random.permutation(self.train_right_x_len)
        test_random_sort = np.random.permutation(self.test_right_x_len)

        train_right_x = np.array(train_right_files)[train_random_sort]
        train_right_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in train_right_x]
        test_right_x = np.array(test_right_files)[test_random_sort]
        test_right_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in test_right_x]

        train_left_x = np.array(train_left_files)[train_random_sort]
        train_left_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in train_left_x]
        test_left_x = np.array(test_left_files)[test_random_sort]
        test_left_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in test_left_x]

        '''리스트 or 배열 텐서화'''
        with tf.variable_scope('file_path_list'):
            self.train_right_x = tf.convert_to_tensor(train_right_x, dtype=tf.string, name='train_right_x')
            self.train_right_y = tf.convert_to_tensor(train_right_y, dtype=tf.int64, name='train_right_y')
            self.test_right_x = tf.convert_to_tensor(test_right_x, dtype=tf.string, name='test_right_x')
            self.test_right_y = tf.convert_to_tensor(test_right_y, dtype=tf.int64, name='test_right_y')

            self.train_left_x = tf.convert_to_tensor(train_left_x, dtype=tf.string, name='train_left_x')
            self.train_left_y = tf.convert_to_tensor(train_left_y, dtype=tf.int64, name='train_left_y')
            self.test_left_x = tf.convert_to_tensor(test_left_x, dtype=tf.string, name='test_left_x')
            self.test_left_y = tf.convert_to_tensor(test_left_y, dtype=tf.int64, name='test_left_y')

    def train_right_normal_data(self, x, y):
        with tf.variable_scope('train_right_normal_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            # x = tf.cast(tf.image.resize_image_with_crop_or_pad(image=x, target_height=50, target_width=100), dtype=tf.float32)
            # x = tf.transpose(x, perm=[1, 0, 2])
            x = tf.image.resize_images(x, size=(100, 50))
            x = tf.divide(x, 255.)
        return x, y

    def train_right_crop_data(self, x, y):
        with tf.variable_scope('train_right_crop_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            # x = tf.cast(tf.image.resize_image_with_crop_or_pad(image=x, target_height=60, target_width=120), dtype=tf.float32)
            # x = tf.transpose(x, perm=[1, 0, 2])
            x = tf.image.resize_images(x, size=(120, 60))
            x = tf.random_crop(value=x, size=(100, 50, 1))
            x = tf.divide(x, 255.)
        return x, y

    def train_left_normal_data(self, x, y):
        with tf.variable_scope('train_left_normal_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            # x = tf.cast(tf.image.resize_image_with_crop_or_pad(image=x, target_height=50, target_width=100), dtype=tf.float32)
            # x = tf.transpose(x, perm=[1, 0, 2])
            x = tf.image.resize_images(x, size=(100, 50))
            x = tf.divide(x, 255.)
        return x, y

    def train_left_crop_data(self, x, y):
        with tf.variable_scope('train_left_crop_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            # x = tf.cast(tf.image.resize_image_with_crop_or_pad(image=x, target_height=60, target_width=120), dtype=tf.float32)
            # x = tf.transpose(x, perm=[1, 0, 2])
            x = tf.image.resize_images(x, size=(120, 60))
            x = tf.random_crop(value=x, size=(100, 50, 1))
            x = tf.divide(x, 255.)
        return x, y

    def dataset_batch_loader(self, dataset, ref_func, name):
        '''
            dataset.map().batch(): 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다.
            dataset_map.make_one_shot_iterator(): 데이터셋을 이터레이터를 통해 지속적으로 불러준다.
            iterator.get_next(): 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다.
        '''
        with tf.variable_scope(name_or_scope=name):
            dataset_map = dataset.map(ref_func).batch(self.batch_size)
            iterator = dataset_map.make_one_shot_iterator()
            batch_input = iterator.get_next()
        return batch_input

    def train_loader(self):
        with tf.variable_scope('train_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_right_x, self.train_right_y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_left_x, self.train_left_y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset,  self.train_right_normal_data, name='right_loader1')
            right_crop_batch1 = self.dataset_batch_loader(right_dataset, self.train_right_crop_data, name='right_loader2')
            right_crop_batch2 = self.dataset_batch_loader(right_dataset, self.train_right_crop_data, name='right_loader3')
            right_crop_batch3 = self.dataset_batch_loader(right_dataset, self.train_right_crop_data, name='right_loader4')
            right_crop_batch4 = self.dataset_batch_loader(right_dataset, self.train_right_crop_data, name='right_loader5')
            right_crop_batch5 = self.dataset_batch_loader(right_dataset, self.train_right_crop_data, name='right_loader6')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.train_left_normal_data, name='left_loader1')
            left_crop_batch1 = self.dataset_batch_loader(left_dataset, self.train_left_crop_data, name='left_loader2')
            left_crop_batch2 = self.dataset_batch_loader(left_dataset, self.train_left_crop_data, name='left_loader3')
            left_crop_batch3 = self.dataset_batch_loader(left_dataset, self.train_left_crop_data, name='left_loader4')
            left_crop_batch4 = self.dataset_batch_loader(left_dataset, self.train_left_crop_data, name='left_loader5')
            left_crop_batch5 = self.dataset_batch_loader(left_dataset, self.train_left_crop_data, name='left_loader6')

        return right_normal_batch, right_crop_batch1, right_crop_batch2, right_crop_batch3, right_crop_batch4, right_crop_batch5,\
               left_normal_batch, left_crop_batch1, left_crop_batch2, left_crop_batch3, left_crop_batch4, left_crop_batch5

    def test_right_normal_data(self, x, y):
        with tf.variable_scope('test_right_normal_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            # x = tf.cast(tf.image.resize_image_with_crop_or_pad(image=x, target_height=50, target_width=100), dtype=tf.float32)
            # x = tf.transpose(x, perm=[1, 0, 2])
            x = tf.image.resize_images(x, size=(100, 50))
            x = tf.divide(x, 255.)
        return x, y

    def test_right_crop_data(self, x, y):
        with tf.variable_scope('test_right_crop_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            # x = tf.cast(tf.image.resize_image_with_crop_or_pad(image=x, target_height=60, target_width=120), dtype=tf.float32)
            # x = tf.transpose(x, perm=[1, 0, 2])
            x = tf.image.resize_images(x, size=(120, 60))
            x = tf.random_crop(value=x, size=(100, 50, 1))
            x = tf.divide(x, 255.)
        return x, y

    def test_left_normal_data(self, x, y):
        with tf.variable_scope('test_left_normal_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            # x = tf.cast(tf.image.resize_image_with_crop_or_pad(image=x, target_height=50, target_width=100), dtype=tf.float32)
            # x = tf.transpose(x, perm=[1, 0, 2])
            x = tf.image.resize_images(x, size=(100, 50))
            x = tf.divide(x, 255.)
        return x, y

    def test_left_crop_data(self, x, y):
        with tf.variable_scope('test_left_crop_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            # x = tf.cast(tf.image.resize_image_with_crop_or_pad(image=x, target_height=60, target_width=120), dtype=tf.float32)
            # x = tf.transpose(x, perm=[1, 0, 2])
            x = tf.image.resize_images(x, size=(120, 60))
            x = tf.random_crop(value=x, size=(100, 50, 1))
            x = tf.divide(x, 255.)
        return x, y

    def test_loader(self):
        with tf.variable_scope('test_loader'):
            # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_right_x, self.test_right_y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_left_x, self.test_left_y)).repeat()
            '''
                dataset.map().batch(): 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다
                dataset_map.make_one_shot_iterator(): 데이터셋을 이터레이터를 통해 지속적으로 불러준다
                iterator.get_next(): 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다 
            '''
            right_normal_dataset_map = right_dataset.map(self.test_right_normal_data).batch(self.batch_size)
            right_normal_iterator = right_normal_dataset_map.make_one_shot_iterator()
            right_normal_batch_input = right_normal_iterator.get_next()

            left_normal_dataset_map = left_dataset.map(self.test_left_normal_data).batch(self.batch_size)
            left_normal_iterator = left_normal_dataset_map.make_one_shot_iterator()
            left_normal_batch_input = left_normal_iterator.get_next()

        return right_normal_batch_input, left_normal_batch_input