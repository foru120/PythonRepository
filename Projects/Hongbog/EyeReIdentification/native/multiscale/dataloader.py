import os
import numpy as np
import re
from collections import defaultdict
import tensorflow as tf
import cv2

class DataLoader:
    LOW_IMG_SIZE = (46, 100)
    MID_IMG_SIZE = (70, 150)
    HIGH_IMG_SIZE = (92, 200)

    def __init__(self, batch_size, train_right_root_path, test_right_root_path, train_left_root_path, test_left_root_path):
        self.batch_size = batch_size
        self.train_right_root_path = train_right_root_path
        self.train_left_root_path = train_left_root_path
        self.test_right_root_path = test_right_root_path
        self.test_left_root_path = test_left_root_path

    def train_init(self):
        train_right_path, train_left_path = defaultdict(list), defaultdict(list)
        train_right_data, train_left_data = defaultdict(list), defaultdict(list)

        '''오른쪽/왼쪽 눈 파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(self.train_right_root_path):
            for file in files:
                full_path = os.path.join(path, file)
                label = int(re.search('.*\\\\(.*)\\\\.*\.png', full_path, re.IGNORECASE).group(1))
                train_right_path[label].append(full_path)

        for (path, dirs, files) in os.walk(self.train_left_root_path):
            for file in files:
                full_path = os.path.join(path, file)
                label = int(re.search('.*\\\\(.*)\\\\.*\.png', full_path, re.IGNORECASE).group(1))
                train_left_path[label].append(full_path)

        '''
            같은/다른 데이터셋 분리
            original -> same: 1200, diff:100
            v2 -> same: 120, diff:100
        '''
        for key in train_right_path.keys():
            '''같은 데이터셋'''
            r_sort = np.random.permutation(len(train_right_path[key]))
            for idx in np.arange(0, 120, 2, dtype=np.int32):
                train_right_data[0].append((train_right_path[key][r_sort[idx]], train_right_path[key][r_sort[idx+1]]))

            l_sort = np.random.permutation(len(train_left_path[key]))
            for idx in np.arange(0, 120, 2, dtype=np.int32):
                train_left_data[0].append((train_left_path[key][l_sort[idx]], train_left_path[key][l_sort[idx+1]]))

            '''다른 데이터셋'''
            for sub_key in train_right_path.keys():
                if key == sub_key:
                    continue

                r_sort1 = np.random.permutation(len(train_right_path[key]))[:100]
                r_sort2 = np.random.permutation(len(train_right_path[sub_key]))[:100]
                for sort_pair in zip(r_sort1, r_sort2):
                    train_right_data[1].append((train_right_path[key][sort_pair[0]], train_right_path[sub_key][sort_pair[1]]))

                l_sort1 = np.random.permutation(len(train_left_path[key]))[:100]
                l_sort2 = np.random.permutation(len(train_left_path[sub_key]))[:100]
                for sort_pair in zip(l_sort1, l_sort2):
                    train_left_data[1].append((train_left_path[key][sort_pair[0]], train_left_path[sub_key][sort_pair[1]]))

        train_right_path, train_left_path = None, None

        '''오른쪽/왼쪽 데이터 정렬 및 X/Y 분리'''
        self.train_tot_len = len(train_right_data[0]) + len(train_right_data[1])
        tot_sort = np.random.permutation(self.train_tot_len)

        train_right_X = train_right_data[0] + train_right_data[1]
        train_right_Y = np.zeros(shape=(len(train_right_data[0]),)).tolist() + np.ones(shape=(len(train_right_data[1]),)).tolist()
        train_right_X = np.asarray(train_right_X)[tot_sort]
        train_right_Y = np.asarray(train_right_Y)[tot_sort]

        train_left_X = train_left_data[0] + train_left_data[1]
        train_left_Y = np.zeros(shape=(len(train_left_data[0]),)).tolist() + np.ones(shape=(len(train_left_data[1]),)).tolist()
        train_left_X = np.asarray(train_left_X)[tot_sort]
        train_left_Y = np.asarray(train_left_Y)[tot_sort]

        '''텐서 변수화'''
        with tf.variable_scope(name_or_scope='train_data_tensor'):
            self.train_right_X = tf.convert_to_tensor(train_right_X, dtype=tf.string, name='train_right_X')
            self.train_right_Y = tf.convert_to_tensor(train_right_Y, dtype=tf.int32, name='train_right_Y')
            self.train_left_X = tf.convert_to_tensor(train_left_X, dtype=tf.string, name='train_left_X')
            self.train_left_Y = tf.convert_to_tensor(train_left_Y, dtype=tf.int32, name='train_left_Y')

    def test_init(self):
        test_right_path, test_left_path = defaultdict(list), defaultdict(list)
        test_right_data, test_left_data = defaultdict(list), defaultdict(list)

        '''오른쪽/왼쪽 눈 파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(self.test_right_root_path):
            for file in files:
                full_path = os.path.join(path, file)
                label = int(re.search('.*\\\\(.*)\\\\.*\.png', full_path, re.IGNORECASE).group(1))
                test_right_path[label].append(full_path)

        for (path, dirs, files) in os.walk(self.test_left_root_path):
            for file in files:
                full_path = os.path.join(path, file)
                label = int(re.search('.*\\\\(.*)\\\\.*\.png', full_path, re.IGNORECASE).group(1))
                test_left_path[label].append(full_path)

        '''같은/다른 데이터셋 분리'''
        for key in test_right_path.keys():
            '''같은 데이터셋'''
            r_sort = np.random.permutation(len(test_right_path[key]))
            for idx in np.arange(0, 600, 2, dtype=np.int32):
                test_right_data[0].append((test_right_path[key][r_sort[idx]], test_right_path[key][r_sort[idx + 1]]))

            l_sort = np.random.permutation(len(test_left_path[key]))
            for idx in np.arange(0, 600, 2, dtype=np.int32):
                test_left_data[0].append((test_left_path[key][l_sort[idx]], test_left_path[key][l_sort[idx + 1]]))

            '''다른 데이터셋'''
            for sub_key in test_right_path.keys():
                if key == sub_key:
                    continue

                r_sort1 = np.random.permutation(len(test_right_path[key]))[:50]
                r_sort2 = np.random.permutation(len(test_right_path[sub_key]))[:50]
                for sort_pair in zip(r_sort1, r_sort2):
                    test_right_data[1].append((test_right_path[key][sort_pair[0]], test_right_path[sub_key][sort_pair[1]]))

                l_sort1 = np.random.permutation(len(test_left_path[key]))[:50]
                l_sort2 = np.random.permutation(len(test_left_path[sub_key]))[:50]
                for sort_pair in zip(l_sort1, l_sort2):
                    test_left_data[1].append((test_left_path[key][sort_pair[0]], test_left_path[sub_key][sort_pair[1]]))

        test_right_path, test_left_path = None, None

        '''오른쪽/왼쪽 데이터 정렬 및 X/Y 분리'''
        self.test_tot_len = len(test_right_data[0]) + len(test_right_data[1])
        tot_sort = np.random.permutation(self.test_tot_len)

        test_right_X = test_right_data[0] + test_right_data[1]
        test_right_Y = np.zeros(shape=(len(test_right_data[0]),)).tolist() + np.ones(shape=(len(test_right_data[1]),)).tolist()
        test_right_X = np.asarray(test_right_X)[tot_sort]
        test_right_Y = np.asarray(test_right_Y)[tot_sort]

        test_left_X = test_left_data[0] + test_left_data[1]
        test_left_Y = np.zeros(shape=(len(test_left_data[0]),)).tolist() + np.ones(shape=(len(test_left_data[1]),)).tolist()
        test_left_X = np.asarray(test_left_X)[tot_sort]
        test_left_Y = np.asarray(test_left_Y)[tot_sort]

        '''텐서 변수화'''
        with tf.variable_scope(name_or_scope='test_data_tensor'):
            self.test_right_X = tf.convert_to_tensor(test_right_X, dtype=tf.string, name='test_right_X')
            self.test_right_Y = tf.convert_to_tensor(test_right_Y, dtype=tf.int32, name='test_right_Y')
            self.test_left_X = tf.convert_to_tensor(test_left_X, dtype=tf.string, name='test_left_X')
            self.test_left_Y = tf.convert_to_tensor(test_left_Y, dtype=tf.int32, name='test_left_Y')

    '''
        훈련 데이터 로딩 부분
    '''
    def tf_equalize_histogram(self, image):
        values_range = tf.constant([0., 255.], dtype=tf.float32)
        histogram = tf.histogram_fixed_width(tf.to_float(image), values_range, 256)
        cdf = tf.cumsum(histogram)
        cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

        img_shape = tf.shape(image)
        pix_cnt = img_shape[-3] * img_shape[-2]
        px_map = tf.round(tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt - 1))
        px_map = tf.cast(px_map, tf.uint8)

        eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
        return eq_hist

    '''
        Low Resolution Augmentation
    '''
    def train_low_normal(self, x, y):
        with tf.variable_scope('train_low_normal'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1, dtype=tf.uint8)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_low_crop(self, x, y):
        with tf.variable_scope('train_low_crop'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1)
            ori_x = tf.image.resize_images(ori_x, size=(60, 120), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.random_crop(value=ori_x, size=(DataLoader.LOW_IMG_SIZE[0], DataLoader.LOW_IMG_SIZE[1], 1))
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=(60, 120), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.random_crop(value=query_x, size=(DataLoader.LOW_IMG_SIZE[0], DataLoader.LOW_IMG_SIZE[1], 1))
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_low_brightness(self, x, y):
        with tf.variable_scope('train_low_brightness'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_brightness(ori_x, max_delta=0.5)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_brightness(query_x, max_delta=0.5)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_low_contrast(self, x, y):
        with tf.variable_scope('train_low_contrast'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_contrast(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_contrast(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_low_hue(self, x, y):
        with tf.variable_scope('train_low_hue'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_hue(ori_x, max_delta=0.08)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_hue(query_x, max_delta=0.08)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_low_saturation(self, x, y):
        with tf.variable_scope('train_low_saturation'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_saturation(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_saturation(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_low_tot_augmentation(self, x, y):
        '''
            Random Crop, Random Brightness, Random Contrast, Random Hue, Random Saturation
        '''
        with tf.variable_scope('train_low_tot_augmentation'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=(56, 120), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.random_crop(value=ori_x, size=(DataLoader.LOW_IMG_SIZE[0], DataLoader.LOW_IMG_SIZE[1], 3))
            ori_x = tf.image.random_brightness(ori_x, max_delta=0.5)
            ori_x = tf.image.random_contrast(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.image.random_hue(ori_x, max_delta=0.08)
            ori_x = tf.image.random_saturation(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[0])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=(56, 120), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.random_crop(value=query_x, size=(DataLoader.LOW_IMG_SIZE[0], DataLoader.LOW_IMG_SIZE[1], 3))
            query_x = tf.image.random_brightness(query_x, max_delta=0.5)
            query_x = tf.image.random_contrast(query_x, lower=0.2, upper=2.0)
            query_x = tf.image.random_hue(query_x, max_delta=0.08)
            query_x = tf.image.random_saturation(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    '''
        Mid Resolution Augmentation
    '''
    def train_mid_normal(self, x, y):
        with tf.variable_scope('train_mid_normal'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_mid_crop(self, x, y):
        with tf.variable_scope('train_mid_crop'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1)
            ori_x = tf.image.resize_images(ori_x, size=(80, 170), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.random_crop(value=ori_x, size=(DataLoader.MID_IMG_SIZE[0], DataLoader.MID_IMG_SIZE[1], 1))
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=(80, 170), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.random_crop(value=query_x, size=(DataLoader.MID_IMG_SIZE[0], DataLoader.MID_IMG_SIZE[1], 1))
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_mid_brightness(self, x, y):
        with tf.variable_scope('train_mid_brightness'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_brightness(ori_x, max_delta=0.5)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_brightness(query_x, max_delta=0.5)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_mid_contrast(self, x, y):
        with tf.variable_scope('train_mid_contrast'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_contrast(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_contrast(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_mid_hue(self, x, y):
        with tf.variable_scope('train_mid_hue'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_hue(ori_x, max_delta=0.08)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_hue(query_x, max_delta=0.08)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_mid_saturation(self, x, y):
        with tf.variable_scope('train_mid_saturation'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_saturation(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_saturation(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_mid_tot_augmentation(self, x, y):
        '''
            Random Crop, Random Brightness, Random Contrast, Random Hue, Random Saturation
        '''
        with tf.variable_scope('train_mid_tot_augmentation'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=(78, 170), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.random_crop(value=ori_x, size=(DataLoader.MID_IMG_SIZE[0], DataLoader.MID_IMG_SIZE[1], 3))
            ori_x = tf.image.random_brightness(ori_x, max_delta=0.5)
            ori_x = tf.image.random_contrast(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.image.random_hue(ori_x, max_delta=0.08)
            ori_x = tf.image.random_saturation(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[0])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=(78, 170), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.random_crop(value=query_x, size=(DataLoader.MID_IMG_SIZE[0], DataLoader.MID_IMG_SIZE[1], 3))
            query_x = tf.image.random_brightness(query_x, max_delta=0.5)
            query_x = tf.image.random_contrast(query_x, lower=0.2, upper=2.0)
            query_x = tf.image.random_hue(query_x, max_delta=0.08)
            query_x = tf.image.random_saturation(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    '''
        High Resolution Augmentation
    '''
    def train_high_normal(self, x, y):
        with tf.variable_scope('train_high_normal'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_high_crop(self, x, y):
        with tf.variable_scope('train_high_crop'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1)
            ori_x = tf.image.resize_images(ori_x, size=(100, 220), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.random_crop(value=ori_x, size=(DataLoader.HIGH_IMG_SIZE[0], DataLoader.HIGH_IMG_SIZE[1], 1))
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=(100, 220), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.random_crop(value=query_x, size=(DataLoader.HIGH_IMG_SIZE[0], DataLoader.HIGH_IMG_SIZE[1], 1))
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_high_brightness(self, x, y):
        with tf.variable_scope('train_high_brightness'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_brightness(ori_x, max_delta=0.5)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_brightness(query_x, max_delta=0.5)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_high_contrast(self, x, y):
        with tf.variable_scope('train_high_contrast'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_contrast(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_contrast(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_high_hue(self, x, y):
        with tf.variable_scope('train_high_hue'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_hue(ori_x, max_delta=0.08)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_hue(query_x, max_delta=0.08)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_high_saturation(self, x, y):
        with tf.variable_scope('train_high_saturation'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.image.random_saturation(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.image.random_saturation(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def train_high_tot_augmentation(self, x, y):
        '''
            Random Crop, Random Brightness, Random Contrast, Random Hue, Random Saturation
        '''
        with tf.variable_scope('train_high_tot_augmentation'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=3)
            ori_x = tf.image.resize_images(ori_x, size=(102, 220), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ori_x = tf.random_crop(value=ori_x, size=(DataLoader.HIGH_IMG_SIZE[0], DataLoader.HIGH_IMG_SIZE[1], 3))
            ori_x = tf.image.random_brightness(ori_x, max_delta=0.5)
            ori_x = tf.image.random_contrast(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.image.random_hue(ori_x, max_delta=0.08)
            ori_x = tf.image.random_saturation(ori_x, lower=0.2, upper=2.0)
            ori_x = tf.cast(ori_x, tf.float32)
            ori_x = tf.clip_by_value(ori_x, clip_value_min=0.0, clip_value_max=255.0)
            ori_x = tf.image.rgb_to_grayscale(ori_x)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[0])
            query_x = tf.image.decode_png(query_x, channels=3)
            query_x = tf.image.resize_images(query_x, size=(102, 220), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            query_x = tf.random_crop(value=query_x, size=(DataLoader.HIGH_IMG_SIZE[0], DataLoader.HIGH_IMG_SIZE[1], 3))
            query_x = tf.image.random_brightness(query_x, max_delta=0.5)
            query_x = tf.image.random_contrast(query_x, lower=0.2, upper=2.0)
            query_x = tf.image.random_hue(query_x, max_delta=0.08)
            query_x = tf.image.random_saturation(query_x, lower=0.2, upper=2.0)
            query_x = tf.cast(query_x, tf.float32)
            query_x = tf.clip_by_value(query_x, clip_value_min=0.0, clip_value_max=255.0)
            query_x = tf.image.rgb_to_grayscale(query_x)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

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

    def train_low_loader(self):
        with tf.variable_scope('train_low_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_right_X, self.train_right_Y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_left_X, self.train_left_Y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.train_low_normal, name='right_normal_loader')
            right_crop_batch = self.dataset_batch_loader(right_dataset, self.train_low_crop, name='right_crop_loader')
            right_brightness_batch = self.dataset_batch_loader(right_dataset, self.train_low_brightness, name='right_brightness_loader')
            right_contrast_batch = self.dataset_batch_loader(right_dataset, self.train_low_contrast, name='right_contrast_loader')
            right_hue_batch = self.dataset_batch_loader(right_dataset, self.train_low_hue, name='right_hue_loader')
            right_saturation_batch = self.dataset_batch_loader(right_dataset, self.train_low_saturation, name='right_saturation_loader')
            right_tot_aug_batch = self.dataset_batch_loader(right_dataset, self.train_low_tot_augmentation, name='right_tot_aug_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.train_low_normal, name='left_normal_loader')
            left_crop_batch = self.dataset_batch_loader(left_dataset, self.train_low_crop, name='left_crop_loader')
            left_brightness_batch = self.dataset_batch_loader(left_dataset, self.train_low_brightness, name='left_brightness_loader')
            left_contrast_batch = self.dataset_batch_loader(left_dataset, self.train_low_contrast, name='left_contrast_loader')
            left_hue_batch = self.dataset_batch_loader(left_dataset, self.train_low_hue, name='left_hue_loader')
            left_saturation_batch = self.dataset_batch_loader(left_dataset, self.train_low_saturation, name='left_saturation_loader')
            left_tot_aug_batch = self.dataset_batch_loader(left_dataset, self.train_low_tot_augmentation, name='left_tot_aug_loader')

        return right_normal_batch, right_crop_batch, right_brightness_batch, right_contrast_batch, right_hue_batch, right_saturation_batch, right_tot_aug_batch, \
               left_normal_batch, left_crop_batch, left_brightness_batch, left_contrast_batch, left_hue_batch, left_saturation_batch, left_tot_aug_batch

    def train_mid_loader(self):
        with tf.variable_scope('train_mid_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_right_X, self.train_right_Y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_left_X, self.train_left_Y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.train_mid_normal, name='right_normal_loader')
            right_crop_batch = self.dataset_batch_loader(right_dataset, self.train_mid_crop, name='right_crop_loader')
            right_brightness_batch = self.dataset_batch_loader(right_dataset, self.train_mid_brightness, name='right_brightness_loader')
            right_contrast_batch = self.dataset_batch_loader(right_dataset, self.train_mid_contrast, name='right_contrast_loader')
            right_hue_batch = self.dataset_batch_loader(right_dataset, self.train_mid_hue, name='right_hue_loader')
            right_saturation_batch = self.dataset_batch_loader(right_dataset, self.train_mid_saturation, name='right_saturation_loader')
            right_tot_aug_batch = self.dataset_batch_loader(right_dataset, self.train_mid_tot_augmentation, name='right_tot_aug_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.train_mid_normal, name='left_normal_loader')
            left_crop_batch = self.dataset_batch_loader(left_dataset, self.train_mid_crop, name='left_crop_loader')
            left_brightness_batch = self.dataset_batch_loader(left_dataset, self.train_mid_brightness, name='left_brightness_loader')
            left_contrast_batch = self.dataset_batch_loader(left_dataset, self.train_mid_contrast, name='left_contrast_loader')
            left_hue_batch = self.dataset_batch_loader(left_dataset, self.train_mid_hue, name='left_hue_loader')
            left_saturation_batch = self.dataset_batch_loader(left_dataset, self.train_mid_saturation, name='left_saturation_loader')
            left_tot_aug_batch = self.dataset_batch_loader(left_dataset, self.train_mid_tot_augmentation, name='left_tot_aug_loader')

        return right_normal_batch, right_crop_batch, right_brightness_batch, right_contrast_batch, right_hue_batch, right_saturation_batch, right_tot_aug_batch, \
               left_normal_batch, left_crop_batch, left_brightness_batch, left_contrast_batch, left_hue_batch, left_saturation_batch, left_tot_aug_batch

    def train_high_loader(self):
        with tf.variable_scope('train_high_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_right_X, self.train_right_Y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_left_X, self.train_left_Y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.train_high_normal, name='right_normal_loader')
            right_crop_batch = self.dataset_batch_loader(right_dataset, self.train_high_crop, name='right_crop_loader')
            right_brightness_batch = self.dataset_batch_loader(right_dataset, self.train_high_brightness, name='right_brightness_loader')
            right_contrast_batch = self.dataset_batch_loader(right_dataset, self.train_high_contrast, name='right_contrast_loader')
            right_hue_batch = self.dataset_batch_loader(right_dataset, self.train_high_hue, name='right_hue_loader')
            right_saturation_batch = self.dataset_batch_loader(right_dataset, self.train_high_saturation, name='right_saturation_loader')
            right_tot_aug_batch = self.dataset_batch_loader(right_dataset, self.train_high_tot_augmentation, name='right_tot_aug_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.train_high_normal, name='left_normal_loader')
            left_crop_batch = self.dataset_batch_loader(left_dataset, self.train_high_crop, name='left_crop_loader')
            left_brightness_batch = self.dataset_batch_loader(left_dataset, self.train_high_brightness, name='left_brightness_loader')
            left_contrast_batch = self.dataset_batch_loader(left_dataset, self.train_high_contrast, name='left_contrast_loader')
            left_hue_batch = self.dataset_batch_loader(left_dataset, self.train_high_hue, name='left_hue_loader')
            left_saturation_batch = self.dataset_batch_loader(left_dataset, self.train_high_saturation, name='left_saturation_loader')
            left_tot_aug_batch = self.dataset_batch_loader(left_dataset, self.train_high_tot_augmentation, name='left_tot_aug_loader')

        return right_normal_batch, right_crop_batch, right_brightness_batch, right_contrast_batch, right_hue_batch, right_saturation_batch, right_tot_aug_batch, \
               left_normal_batch, left_crop_batch, left_brightness_batch, left_contrast_batch, left_hue_batch, left_saturation_batch, left_tot_aug_batch

    '''
        테스트 데이터 로딩 부분
    '''

    def test_low_normal(self, x, y):
        with tf.variable_scope('test_low_normal'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def test_mid_normal(self, x, y):
        with tf.variable_scope('test_mid_normal'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def test_high_normal(self, x, y):
        with tf.variable_scope('test_high_normal'):
            ori_x = tf.read_file(x[0])
            ori_x = tf.image.decode_png(ori_x, channels=1)
            ori_x = tf.image.resize_images(ori_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # ori_x = self.tf_equalize_histogram(ori_x)
            ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

            query_x = tf.read_file(x[1])
            query_x = tf.image.decode_png(query_x, channels=1)
            query_x = tf.image.resize_images(query_x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # query_x = self.tf_equalize_histogram(query_x)
            query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
        return ori_x, query_x, y

    def test_low_loader(self):
        with tf.variable_scope('test_low_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_right_X, self.test_right_Y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_left_X, self.test_left_Y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.test_low_normal, name='right_normal_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.test_low_normal, name='left_normal_loader')

        return right_normal_batch, left_normal_batch

    def test_mid_loader(self):
        with tf.variable_scope('test_mid_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_right_X, self.test_right_Y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_left_X, self.test_left_Y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.test_mid_normal, name='right_normal_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.test_mid_normal, name='left_normal_loader')

        return right_normal_batch, left_normal_batch

    def test_high_loader(self):
        with tf.variable_scope('test_high_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_right_X, self.test_right_Y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_left_X, self.test_left_Y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.test_high_normal, name='right_normal_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.test_high_normal, name='left_normal_loader')

        return right_normal_batch, left_normal_batch