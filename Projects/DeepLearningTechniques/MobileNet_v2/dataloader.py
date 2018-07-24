import tensorflow as tf
import os
import numpy as np
import re

class DataLoader:
    def __init__(self, batch_size, data_path):
        self.batch_size = batch_size

        train_x, test_x, tot_x = None, None, []

        '''파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(data_path):
            for file in files:
                tot_x.append(os.path.join(path, file))

        tot_x = np.asarray(tot_x)
        tot_len = tot_x.shape[0]
        random_sort = np.random.permutation(tot_len)

        '''Train/Test Dataset 분리'''
        train_x = tot_x[random_sort[: int(tot_len * 0.8)]]
        train_y = [int(re.search('.*\\\\(\d)\_.*\.jpg', x, re.IGNORECASE).group(1)) for x in train_x]
        test_x = tot_x[int(tot_len * 0.8):]
        test_y = [int(re.search('.*\\\\(\d)\_.*\.jpg', x, re.IGNORECASE).group(1)) for x in test_x]

        self.train_len = train_x.shape[0]
        self.test_len = test_x.shape[0]

        '''리스트 or 배열 텐서화'''
        with tf.variable_scope('file_path_list'):
            self.train_x = tf.convert_to_tensor(train_x, dtype=tf.string, name='train_x')
            self.train_y = tf.convert_to_tensor(train_y, dtype=tf.int32, name='train_y')
            self.test_x = tf.convert_to_tensor(test_x, dtype=tf.string, name='test_x')
            self.test_y = tf.convert_to_tensor(test_y, dtype=tf.int32, name='test_y')

    def train_normal(self, x, y):
        with tf.variable_scope('train_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=(224, 224))
            x = tf.divide(x, 255.)
        return x, y

    def train_random_crop(self, x, y):
        with tf.variable_scope('train_random_crop'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=(250, 250))
            x = tf.random_crop(value=x, size=(224, 224, 3))
            x = tf.divide(x, 255.)
        return x, y

    def train_random_brightness(self, x, y):
        with tf.variable_scope('train_random_brightness'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=(224, 224))
            x = tf.image.random_brightness(x, max_delta=0.5)
            x = tf.divide(x, 255.)
        return x, y

    def train_random_contrast(self, x, y):
        with tf.variable_scope('train_random_contrast'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=(224, 224))
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.divide(x, 255.)
        return x, y

    def train_random_hue(self, x, y):
        with tf.variable_scope('train_random_hue'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=(224, 224))
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.divide(x, 255.)
        return x, y

    def train_random_saturation(self, x, y):
        with tf.variable_scope('train_random_saturation'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=(224, 224))
            x = tf.image.random_saturation(x,lower=0.2, upper=2.0)
            x = tf.divide(x, 255.)
        return x, y

    def dataset_batch_loader(self, dataset, ref_func, name):
        '''
            dataset.map().batch(): 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에
                                   넣어준다.
            dataset_map.make_one_shot_iterator(): 데이터셋을 이터레이터를 통해 지속적으로 불러준다.
            iterator.get_next(): 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을
                                 불러온다.
        '''
        with tf.variable_scope(name_or_scope=name):
            dataset_map = dataset.map(ref_func).batch(self.batch_size)
            iterator = dataset_map.make_one_shot_iterator()
            batch_input = iterator.get_next()
        return batch_input

    def train_loader(self):
        with tf.variable_scope('train_loader'):
            '''
                데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 
                반복적으로 사용한다.
            '''
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).repeat()

            normal_img = self.dataset_batch_loader(dataset,  self.train_normal, name='train_loader1')
            random_crop_img = self.dataset_batch_loader(dataset, self.train_random_crop, name='train_loader2')
            random_bri_img = self.dataset_batch_loader(dataset, self.train_random_brightness, name='train_loader3')
            random_cont_img = self.dataset_batch_loader(dataset, self.train_random_contrast, name='train_loader4')
            random_hue_img = self.dataset_batch_loader(dataset, self.train_random_hue, name='train_loader5')
            random_sat_img = self.dataset_batch_loader(dataset, self.train_random_saturation, name='train_loader6')

        return normal_img, random_crop_img, random_bri_img, random_cont_img, random_hue_img, random_sat_img

    def test_normal(self, x, y):
        with tf.variable_scope('test_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=(224, 224))
            x = tf.divide(x, 255.)
        return x, y

    def test_loader(self):
        with tf.variable_scope('test_loader'):
            '''
                데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 
                반복적으로 사용한다.
            '''
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_x, self.test_y)).repeat()

            normal_img = self.dataset_batch_loader(dataset, self.test_normal, name='test_loader1')

        return normal_img