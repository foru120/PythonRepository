import os
import re
import numpy as np

from Projects.Hongbog.MultiGPU.constants import *

class DataLoader:
    # todo train/test => (클래스 당 5000/1000)

    def __init__(self):
        self.image_width = flags.FLAGS.image_width
        self.image_height = flags.FLAGS.image_height
        self.batch_size = flags.FLAGS.batch_size
        self.data_path = flags.FLAGS.data_path

    def init_train(self):
        train_x, train_y = [], []

        for (path, dirs, files) in os.walk(os.path.join(self.data_path, 'train')):
            for file in files:
                train_x.append(os.path.join(path, file))
                train_y.append(re.match('.*\\\(\d+)', path).group(1))

        self.train_len = len(train_y)

        #todo train data random sort
        random_sort = np.random.permutation(self.train_len)

        train_x, train_y = np.asarray(train_x, dtype=np.string_)[random_sort], np.asarray(train_y, dtype=np.int64)[random_sort]

        #todo (Numpy / List) => Tensor 로 변환
        with tf.variable_scope(name_or_scope='data_tensor'):
            self.train_x = tf.convert_to_tensor(value=train_x, dtype=tf.string, name='train_x')
            self.train_y = tf.convert_to_tensor(value=train_y, dtype=tf.int64, name='train_y')

    def init_test(self):
        test_x, test_y = [], []

        for (path, dirs, files) in os.walk(os.path.join(self.data_path, 'test')):
            for file in files:
                test_x.append(os.path.join(path, file))
                test_y.append(re.match('.*\\\(\d+)', path).group(1))

        self.test_len = len(test_y)

        #todo test data random sort
        random_sort = np.random.permutation(self.test_len)

        test_x, test_y = np.asarray(test_x, dtype=np.string_)[random_sort], np.asarray(test_y, dtype=np.int64)[random_sort]

        #todo (Numpy / List) -> Tensor 로 변환
        with tf.variable_scope(name_or_scope='data_tensor'):
            self.test_x = tf.convert_to_tensor(value=test_x, dtype=tf.string, name='test_x')
            self.test_y = tf.convert_to_tensor(value=test_y, dtype=tf.int64, name='test_y')

    def distorted_image(self, x, y):
        with tf.variable_scope(name_or_scope='distorted_image'):
            x = tf.read_file(filename=x)
            x = tf.image.decode_png(contents=x, channels=3, name='decode_png')
            x = tf.image.resize_images(images=x, size=(self.image_height + 5, self.image_width + 5), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.random_crop(value=x, size=(self.image_height, self.image_width, 3))
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, max_delta=63)
            x = tf.image.random_contrast(x, lower=0.2, upper=1.8)
            x = tf.image.per_image_standardization(x)
        return x, y

    def normal_image(self, x, y):
        with tf.variable_scope(name_or_scope='normal_image'):
            x = tf.read_file(filename=x)
            x = tf.image.decode_png(contents=x, channels=3, name='decode_png')
            x = tf.image.per_image_standardization(x)
        return x, y

    def dataset_batch_loader(self, dataset, ref_func, name):
        with tf.variable_scope(name_or_scope=name):
            dataset_map = dataset.map(ref_func).batch(self.batch_size)
            iterator = dataset_map.make_one_shot_iterator()
            x, y = iterator.get_next()
        return x, y

    def train_loader(self):
        with tf.variable_scope('train_loader'):
            '''
                repeat(): 데이터셋이 끝에 도달했을 때 다시 처음부터 수행하게 하는 함수
                shuffle(): 데이터셋에 대해 random sort 기능을 수행하는 함수 (괄호안에 값이 전체 데이터 수보다 크면 전체 데이터에 대한 random sort)
            '''
            dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).repeat()

            x, y = self.dataset_batch_loader(dataset, self.distorted_image, name='distorted_batch')

        return x, y

    def test_loader(self):
        with tf.variable_scope('test_loader'):
            dataset = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y)).repeat()

            x, y = self.dataset_batch_loader(dataset, self.normal_image, name='normal_batch')

        return x, y