import tensorflow as tf
import os
import numpy as np
import re

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
        train_right_files, train_left_files = [], []

        '''오른쪽/왼쪽 눈 파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(self.train_right_root_path):
            for file in files:
                train_right_files.append(os.path.join(path, file))

        for (path, dirs, files) in os.walk(self.train_left_root_path):
            for file in files:
                train_left_files.append(os.path.join(path, file))

        '''오른쪽/왼쪽 눈 파일 개수'''
        self.train_right_x_len = len(train_right_files)
        self.train_left_x_len = len(train_left_files)

        '''오른쪽/왼쪽 눈 x, y 데이터 설정'''
        train_random_sort = np.random.permutation(self.train_right_x_len)

        train_right_x = np.asarray(train_right_files)[train_random_sort]
        train_right_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in train_right_x]
        train_left_x = np.asarray(train_left_files)[train_random_sort]
        train_left_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in train_left_x]

        '''리스트 or 배열 텐서화'''
        with tf.variable_scope('train_path_tensor'):
            self.train_right_x = tf.convert_to_tensor(train_right_x, dtype=tf.string, name='train_right_x')
            self.train_right_y = tf.convert_to_tensor(train_right_y, dtype=tf.int64, name='train_right_y')
            self.train_left_x = tf.convert_to_tensor(train_left_x, dtype=tf.string, name='train_left_x')
            self.train_left_y = tf.convert_to_tensor(train_left_y, dtype=tf.int64, name='train_left_y')

    def test_init(self):
        test_right_files, test_left_files = [], []

        '''오른쪽/왼쪽 눈 파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(self.test_right_root_path):
            for file in files:
                test_right_files.append(os.path.join(path, file))

        for (path, dirs, files) in os.walk(self.test_left_root_path):
            for file in files:
                test_left_files.append(os.path.join(path, file))

        '''오른쪽/왼쪽 눈 파일 개수'''
        self.test_right_x_len = len(test_right_files)
        self.test_left_x_len = len(test_left_files)

        '''오른쪽/왼쪽 눈 x, y 데이터 설정'''
        test_random_sort = np.random.permutation(self.test_right_x_len)

        test_right_x = np.asarray(test_right_files)[test_random_sort]
        test_right_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in test_right_x]
        test_left_x = np.asarray(test_left_files)[test_random_sort]
        test_left_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in test_left_x]

        '''리스트 or 배열 텐서화'''
        with tf.variable_scope('test_path_tensor'):
            self.test_right_x = tf.convert_to_tensor(test_right_x, dtype=tf.string, name='test_right_x')
            self.test_right_y = tf.convert_to_tensor(test_right_y, dtype=tf.int64, name='test_right_y')
            self.test_left_x = tf.convert_to_tensor(test_left_x, dtype=tf.string, name='test_left_x')
            self.test_left_y = tf.convert_to_tensor(test_left_y, dtype=tf.int64, name='test_left_y')

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
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_crop(self, x, y):
        with tf.variable_scope('train_low_crop'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=(60, 120), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.random_crop(value=x, size=(DataLoader.LOW_IMG_SIZE[0], DataLoader.LOW_IMG_SIZE[1], 1))
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_brightness(self, x, y):
        with tf.variable_scope('train_low_brightness'):
            '''
                tf.image.random_brightness 에서 max_delta 값은 [0, 1) 사이 값으로 지정하고,
                random 으로 -max_delta ~ +max_delta 사이의 값을 모든 이미지 픽셀 값에 더한다.
                이때, 기존 이미지를 [0, 1) 사이 값으로 변경한 후 값을 추가한다.
            '''
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_brightness(x, max_delta=0.5)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_contrast(self, x, y):
        with tf.variable_scope('train_low_contrast'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_hue(self, x, y):
        with tf.variable_scope('train_low_hue'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_saturation(self, x, y):
        with tf.variable_scope('train_low_saturation'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_saturation(x,lower=0.2, upper=2.0)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    '''
        Mid Resolution Augmentation
    '''
    def train_mid_normal(self, x, y):
        with tf.variable_scope('train_mid_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_crop(self, x, y):
        with tf.variable_scope('train_mid_crop'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=(80, 170), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.random_crop(value=x, size=(DataLoader.MID_IMG_SIZE[0], DataLoader.MID_IMG_SIZE[1], 1))
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_brightness(self, x, y):
        with tf.variable_scope('train_mid_brightness'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_brightness(x, max_delta=0.5)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_contrast(self, x, y):
        with tf.variable_scope('train_mid_contrast'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_hue(self, x, y):
        with tf.variable_scope('train_mid_hue'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_saturation(self, x, y):
        with tf.variable_scope('train_mid_saturation'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_saturation(x, lower=0.2, upper=2.0)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    '''
        High Resolution Augmentation
    '''
    def train_high_normal(self, x, y):
        with tf.variable_scope('train_high_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_crop(self, x, y):
        with tf.variable_scope('train_high_crop'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=(100, 220), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.random_crop(value=x, size=(DataLoader.HIGH_IMG_SIZE[0], DataLoader.HIGH_IMG_SIZE[1], 1))
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_brightness(self, x, y):
        with tf.variable_scope('train_high_brightness'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_brightness(x, max_delta=0.5)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_contrast(self, x, y):
        with tf.variable_scope('train_high_contrast'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_hue(self, x, y):
        with tf.variable_scope('train_high_hue'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_saturation(self, x, y):
        with tf.variable_scope('train_high_saturation'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_saturation(x, lower=0.2, upper=2.0)
            x = tf.cast(x, tf.float32)
            x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=255.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
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

    def train_low_loader(self):
        with tf.variable_scope('train_low_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_right_x, self.train_right_y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_left_x, self.train_left_y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.train_low_normal, name='right_normal_loader')
            right_crop_batch = self.dataset_batch_loader(right_dataset, self.train_low_crop, name='right_crop_loader')
            right_brightness_batch = self.dataset_batch_loader(right_dataset, self.train_low_brightness, name='right_brightness_loader')
            right_contrast_batch = self.dataset_batch_loader(right_dataset, self.train_low_contrast, name='right_contrast_loader')
            right_hue_batch = self.dataset_batch_loader(right_dataset, self.train_low_hue, name='right_hue_loader')
            right_saturation_batch = self.dataset_batch_loader(right_dataset, self.train_low_saturation, name='right_saturation_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.train_low_normal, name='left_normal_loader')
            left_crop_batch = self.dataset_batch_loader(left_dataset, self.train_low_crop, name='left_crop_loader')
            left_brightness_batch = self.dataset_batch_loader(left_dataset, self.train_low_brightness, name='left_brightness_loader')
            left_contrast_batch = self.dataset_batch_loader(left_dataset, self.train_low_contrast, name='left_contrast_loader')
            left_hue_batch = self.dataset_batch_loader(left_dataset, self.train_low_hue, name='left_hue_loader')
            left_saturation_batch = self.dataset_batch_loader(left_dataset, self.train_low_saturation, name='left_saturation_loader')

        return right_normal_batch, right_crop_batch, right_brightness_batch, right_contrast_batch, right_hue_batch, right_saturation_batch, \
               left_normal_batch, left_crop_batch, left_brightness_batch, left_contrast_batch, left_hue_batch, left_saturation_batch

    def train_mid_loader(self):
        with tf.variable_scope('train_mid_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_right_x, self.train_right_y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_left_x, self.train_left_y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.train_mid_normal, name='right_normal_loader')
            right_crop_batch = self.dataset_batch_loader(right_dataset, self.train_mid_crop, name='right_crop_loader')
            right_brightness_batch = self.dataset_batch_loader(right_dataset, self.train_mid_brightness, name='right_brightness_loader')
            right_contrast_batch = self.dataset_batch_loader(right_dataset, self.train_mid_contrast, name='right_contrast_loader')
            right_hue_batch = self.dataset_batch_loader(right_dataset, self.train_mid_hue, name='right_hue_loader')
            right_saturation_batch = self.dataset_batch_loader(right_dataset, self.train_mid_saturation, name='right_saturation_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.train_mid_normal, name='left_normal_loader')
            left_crop_batch = self.dataset_batch_loader(left_dataset, self.train_mid_crop, name='left_crop_loader')
            left_brightness_batch = self.dataset_batch_loader(left_dataset, self.train_mid_brightness, name='left_brightness_loader')
            left_contrast_batch = self.dataset_batch_loader(left_dataset, self.train_mid_contrast, name='left_contrast_loader')
            left_hue_batch = self.dataset_batch_loader(left_dataset, self.train_mid_hue, name='left_hue_loader')
            left_saturation_batch = self.dataset_batch_loader(left_dataset, self.train_mid_saturation, name='left_saturation_loader')

        return right_normal_batch, right_crop_batch, right_brightness_batch, right_contrast_batch, right_hue_batch, right_saturation_batch, \
               left_normal_batch, left_crop_batch, left_brightness_batch, left_contrast_batch, left_hue_batch, left_saturation_batch

    def train_high_loader(self):
        with tf.variable_scope('train_high_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_right_x, self.train_right_y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_left_x, self.train_left_y)).repeat()

            right_normal_batch = self.dataset_batch_loader(right_dataset, self.train_high_normal, name='right_normal_loader')
            right_crop_batch = self.dataset_batch_loader(right_dataset, self.train_high_crop, name='right_crop_loader')
            right_brightness_batch = self.dataset_batch_loader(right_dataset, self.train_high_brightness, name='right_brightness_loader')
            right_contrast_batch = self.dataset_batch_loader(right_dataset, self.train_high_contrast, name='right_contrast_loader')
            right_hue_batch = self.dataset_batch_loader(right_dataset, self.train_high_hue, name='right_hue_loader')
            right_saturation_batch = self.dataset_batch_loader(right_dataset, self.train_high_saturation, name='right_saturation_loader')

            left_normal_batch = self.dataset_batch_loader(left_dataset, self.train_high_normal, name='left_normal_loader')
            left_crop_batch = self.dataset_batch_loader(left_dataset, self.train_high_crop, name='left_crop_loader')
            left_brightness_batch = self.dataset_batch_loader(left_dataset, self.train_high_brightness, name='left_brightness_loader')
            left_contrast_batch = self.dataset_batch_loader(left_dataset, self.train_high_contrast, name='left_contrast_loader')
            left_hue_batch = self.dataset_batch_loader(left_dataset, self.train_high_hue, name='left_hue_loader')
            left_saturation_batch = self.dataset_batch_loader(left_dataset, self.train_high_saturation, name='left_saturation_loader')

        return right_normal_batch, right_crop_batch, right_brightness_batch, right_contrast_batch, right_hue_batch, right_saturation_batch, \
               left_normal_batch, left_crop_batch, left_brightness_batch, left_contrast_batch, left_hue_batch, left_saturation_batch

    def test_low_normal(self, x, y):
        with tf.variable_scope('test_low_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def test_mid_normal(self, x, y):
        with tf.variable_scope('test_mid_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def test_high_normal(self, x, y):
        with tf.variable_scope('test_high_normal_data'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def test_low_loader(self):
        with tf.variable_scope('test_low_loader'):
            # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_right_x, self.test_right_y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_left_x, self.test_left_y)).repeat()
            '''
                dataset.map().batch(): 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다
                dataset_map.make_one_shot_iterator(): 데이터셋을 이터레이터를 통해 지속적으로 불러준다
                iterator.get_next(): 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다 
            '''
            right_normal_dataset_map = right_dataset.map(self.test_low_normal).batch(self.batch_size)
            right_normal_iterator = right_normal_dataset_map.make_one_shot_iterator()
            right_normal_batch = right_normal_iterator.get_next()

            left_normal_dataset_map = left_dataset.map(self.test_low_normal).batch(self.batch_size)
            left_normal_iterator = left_normal_dataset_map.make_one_shot_iterator()
            left_normal_batch = left_normal_iterator.get_next()

        return right_normal_batch, left_normal_batch

    def test_mid_loader(self):
        with tf.variable_scope('test_mid_loader'):
            # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_right_x, self.test_right_y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_left_x, self.test_left_y)).repeat()
            '''
                dataset.map().batch(): 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다
                dataset_map.make_one_shot_iterator(): 데이터셋을 이터레이터를 통해 지속적으로 불러준다
                iterator.get_next(): 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다 
            '''
            right_normal_dataset_map = right_dataset.map(self.test_mid_normal).batch(self.batch_size)
            right_normal_iterator = right_normal_dataset_map.make_one_shot_iterator()
            right_normal_batch = right_normal_iterator.get_next()

            left_normal_dataset_map = left_dataset.map(self.test_mid_normal).batch(self.batch_size)
            left_normal_iterator = left_normal_dataset_map.make_one_shot_iterator()
            left_normal_batch = left_normal_iterator.get_next()

        return right_normal_batch, left_normal_batch

    def test_high_loader(self):
        with tf.variable_scope('test_high_loader'):
            # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
            right_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_right_x, self.test_right_y)).repeat()
            left_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_left_x, self.test_left_y)).repeat()
            '''
                dataset.map().batch(): 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다
                dataset_map.make_one_shot_iterator(): 데이터셋을 이터레이터를 통해 지속적으로 불러준다
                iterator.get_next(): 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다 
            '''
            right_normal_dataset_map = right_dataset.map(self.test_high_normal).batch(self.batch_size)
            right_normal_iterator = right_normal_dataset_map.make_one_shot_iterator()
            right_normal_batch = right_normal_iterator.get_next()

            left_normal_dataset_map = left_dataset.map(self.test_high_normal).batch(self.batch_size)
            left_normal_iterator = left_normal_dataset_map.make_one_shot_iterator()
            left_normal_batch = left_normal_iterator.get_next()

        return right_normal_batch, left_normal_batch