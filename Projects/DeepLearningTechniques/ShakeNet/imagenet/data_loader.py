import os
import re
import numpy as np
from xml.etree.ElementTree import parse

from Projects.DeepLearningTechniques.ShakeNet.imagenet.constants import *

class DataLoader:
    #todo train/test/validation => (클래스 당 1300//)

    def __init__(self):
        self.image_width = flags.FLAGS.image_width
        self.image_height = flags.FLAGS.image_height
        self.batch_size = flags.FLAGS.batch_size
        self.data_path = flags.FLAGS.data_path
        self.data_anno_path = flags.FLAGS.data_anno_path

        self.init_class()
        self.init_annotation()

    def init_class(self):
        self.cls = {}

        cnt = 0
        for dir in os.listdir(os.path.join(self.data_path, 'train')):
            self.cls[dir] = cnt
            cnt += 1

    def get_anno_info(self, file_path):
        tree = parse(file_path)
        root = tree.getroot()

        filename = root.find('filename').text
        name = root.find('object').find('name').text

        return filename, name

    def init_annotation(self):
        self.anno = {}

        # todo validation data annotation setting
        for (path, dirs, files) in os.walk(os.path.join(self.data_anno_path, 'val')):
            for file in files:
                filename, name = self.get_anno_info(os.path.join(path, file))
                self.anno[filename] = self.cls[name]

    def init_train(self):
        train_x, train_y = [], []

        for (path, dirs, files) in os.walk(os.path.join(self.data_path, 'train')):
            for file in files:
                train_x.append(os.path.join(path, file))
                train_y.append(self.cls[re.match('.*\\\(n[\d]+)', path).group(1)])

        self.train_len = len(train_y)

        #todo train data random sort
        random_sort = np.random.permutation(self.train_len)

        train_x, train_y = np.asarray(train_x, dtype=np.string_)[random_sort], np.asarray(train_y, dtype=np.int64)[random_sort]

        #todo (Numpy / List) => Tensor 로 변환
        with tf.variable_scope(name_or_scope='data_tensor'):
            self.train_x = tf.convert_to_tensor(value=train_x, dtype=tf.string, name='train_x')
            self.train_y = tf.convert_to_tensor(value=train_y, dtype=tf.int64, name='train_y')

    def init_validation(self):
        valid_x, valid_y = [], []

        for (path, dirs, files) in os.walk(os.path.join(self.data_path, 'val')):
            for file in files:
                valid_x.append(os.path.join(path, file))
                valid_y.append(self.anno[os.path.splitext(file)[0]])

        self.valid_len = len(valid_y)

        #todo validation data random sort
        random_sort = np.random.permutation(self.valid_len)

        valid_x, valid_y = np.asarray(valid_x, dtype=np.string_)[random_sort], np.asarray(valid_y, dtype=np.int64)[random_sort]

        #todo (Numpy / List) => Tensor 로 변환
        with tf.variable_scope(name_or_scope='data_tensor'):
            self.valid_x = tf.convert_to_tensor(value=valid_x, dtype=tf.string, name='valid_x')
            self.valid_y = tf.convert_to_tensor(value=valid_y, dtype=tf.int64, name='valid_y')

    def train_normal(self, x, y):
        with tf.variable_scope(name_or_scope='train_normal'):
            x = tf.read_file(filename=x)
            x = tf.image.decode_jpeg(contents=x, channels=3, name='decode_jpeg')
            x = tf.image.resize_images(images=x, size=(self.image_height, self.image_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_random_crop(self, x, y):
        with tf.variable_scope(name_or_scope='train_random_crop'):
            x = tf.read_file(filename=x)
            x = tf.image.decode_jpeg(contents=x, channels=3, name='decode_jpeg')
            x = tf.image.resize_images(images=x, size=(self.image_height+20, self.image_width+20), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.random_crop(value=x, size=(self.image_height, self.image_width, 3))
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def valid_normal(self, x, y):
        with tf.variable_scope(name_or_scope='valid_normal'):
            x = tf.read_file(filename=x)
            x = tf.image.decode_jpeg(contents=x, channels=3, name='decode_jpeg')
            x = tf.image.resize_images(images=x, size=(self.image_height, self.image_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def dataset_batch_loader(self, dataset, ref_func, name):
        with tf.variable_scope(name_or_scope=name):
            dataset_map = dataset.map(ref_func).batch(self.batch_size)
            iterator = dataset_map.make_one_shot_iterator()
            batch_input = iterator.get_next()
        return batch_input

    def train_loader(self):
        with tf.variable_scope('train_loader'):
            '''
                repeat(): 데이터셋이 끝에 도달했을 때 다시 처음부터 수행하게 하는 함수
                shuffle(): 데이터셋에 대해 random sort 기능을 수행하는 함수 (괄호안에 값이 전체 데이터 수보다 크면 전체 데이터에 대한 random sort)
            '''
            dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).repeat()

            normal_batch = self.dataset_batch_loader(dataset, self.train_normal, name='normal_batch')
            random_crop_batch = self.dataset_batch_loader(dataset, self.train_random_crop, name='random_crop_batch')

        return normal_batch, random_crop_batch

    def valid_loader(self):
        with tf.variable_scope('valid_loader'):
            dataset = tf.data.Dataset.from_tensor_slices((self.valid_x, self.valid_y)).repeat()

            normal_batch = self.dataset_batch_loader(dataset, self.valid_normal, name='normal_batch')

        return normal_batch