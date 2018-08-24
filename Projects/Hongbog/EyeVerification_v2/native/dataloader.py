import tensorflow as tf
import os
import numpy as np
import re

class DataLoader:
    LOW_IMG_SIZE = (36, 200)
    MID_IMG_SIZE = (48, 260)
    HIGH_IMG_SIZE = (60, 320)

    def __init__(self, batch_size, train_root_path, test_root_path):
        self.batch_size = batch_size
        self.train_root_path = train_root_path
        self.test_root_path = test_root_path

    def train_init(self):
        train_files = []

        '''오른쪽/왼쪽 눈 파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(self.train_root_path):
            for file in files:
                train_files.append(os.path.join(path, file))

        '''오른쪽/왼쪽 눈 파일 개수'''
        self.train_tot_len = len(train_files)

        '''오른쪽/왼쪽 눈 x, y 데이터 설정'''
        train_random_sort = np.random.permutation(self.train_tot_len)

        train_x = np.asarray(train_files)[train_random_sort]
        train_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in train_x]

        '''리스트 or 배열 텐서화'''
        with tf.variable_scope('train_data_tensor'):
            self.train_x = tf.convert_to_tensor(train_x, dtype=tf.string, name='train_x')
            self.train_y = tf.convert_to_tensor(train_y, dtype=tf.int64, name='train_y')

    def test_init(self):
        test_files = []

        '''오른쪽/왼쪽 눈 파일 경로 불러오기'''
        for (path, dirs, files) in os.walk(self.test_root_path):
            for file in files:
                test_files.append(os.path.join(path, file))

        '''오른쪽/왼쪽 눈 파일 개수'''
        self.test_tot_len = len(test_files)

        '''오른쪽/왼쪽 눈 x, y 데이터 설정'''
        test_random_sort = np.random.permutation(self.test_tot_len)

        test_x = np.asarray(test_files)[test_random_sort]
        test_y = [int(re.search('.*\\\\(.*)\\\\.*\.png', x, re.IGNORECASE).group(1)) for x in test_x]

        '''리스트 or 배열 텐서화'''
        with tf.variable_scope('test_data_tensor'):
            self.test_x = tf.convert_to_tensor(test_x, dtype=tf.string, name='test_x')
            self.test_y = tf.convert_to_tensor(test_y, dtype=tf.int64, name='test_y')

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
            x = tf.image.resize_images(x, size=(50, 220), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.random_crop(value=x, size=(DataLoader.LOW_IMG_SIZE[0], DataLoader.LOW_IMG_SIZE[1], 1))
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_brightness(self, x, y):
        with tf.variable_scope('train_low_brightness'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_brightness(x, max_delta=0.5)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_contrast(self, x, y):
        with tf.variable_scope('train_low_contrast'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_hue(self, x, y):
        with tf.variable_scope('train_low_hue'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_low_saturation(self, x, y):
        with tf.variable_scope('train_low_saturation'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_saturation(x,lower=0.2, upper=2.0)
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
            x = tf.image.resize_images(x, size=(60, 280), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.random_crop(value=x, size=(DataLoader.MID_IMG_SIZE[0], DataLoader.MID_IMG_SIZE[1], 1))
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_brightness(self, x, y):
        with tf.variable_scope('train_mid_brightness'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_brightness(x, max_delta=0.5)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_contrast(self, x, y):
        with tf.variable_scope('train_mid_contrast'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_hue(self, x, y):
        with tf.variable_scope('train_mid_hue'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_mid_saturation(self, x, y):
        with tf.variable_scope('train_mid_saturation'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_saturation(x, lower=0.2, upper=2.0)
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
            x = tf.image.resize_images(x, size=(70, 340), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.random_crop(value=x, size=(DataLoader.HIGH_IMG_SIZE[0], DataLoader.HIGH_IMG_SIZE[1], 1))
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_brightness(self, x, y):
        with tf.variable_scope('train_high_brightness'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_brightness(x, max_delta=0.5)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_contrast(self, x, y):
        with tf.variable_scope('train_high_contrast'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_hue(self, x, y):
        with tf.variable_scope('train_high_hue'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.image.rgb_to_grayscale(x)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def train_high_saturation(self, x, y):
        with tf.variable_scope('train_high_saturation'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.image.random_saturation(x, lower=0.2, upper=2.0)
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
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).repeat()

            normal_batch = self.dataset_batch_loader(dataset, self.train_low_normal, name='normal_loader')
            crop_batch = self.dataset_batch_loader(dataset, self.train_low_crop, name='crop_loader')
            brightness_batch = self.dataset_batch_loader(dataset, self.train_low_brightness, name='brightness_loader')
            contrast_batch = self.dataset_batch_loader(dataset, self.train_low_contrast, name='contrast_loader')
            hue_batch = self.dataset_batch_loader(dataset, self.train_low_hue, name='hue_loader')
            saturation_batch = self.dataset_batch_loader(dataset, self.train_low_saturation, name='saturation_loader')

        return normal_batch, crop_batch, brightness_batch, contrast_batch, hue_batch, saturation_batch

    def train_mid_loader(self):
        with tf.variable_scope('train_mid_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).repeat()

            normal_batch = self.dataset_batch_loader(dataset, self.train_mid_normal, name='normal_loader')
            crop_batch = self.dataset_batch_loader(dataset, self.train_mid_crop, name='crop_loader')
            brightness_batch = self.dataset_batch_loader(dataset, self.train_mid_brightness, name='brightness_loader')
            contrast_batch = self.dataset_batch_loader(dataset, self.train_mid_contrast, name='contrast_loader')
            hue_batch = self.dataset_batch_loader(dataset, self.train_mid_hue, name='hue_loader')
            saturation_batch = self.dataset_batch_loader(dataset, self.train_mid_saturation, name='saturation_loader')

        return normal_batch, crop_batch, brightness_batch, contrast_batch, hue_batch, saturation_batch

    def train_high_loader(self):
        with tf.variable_scope('train_high_loader'):
            '''데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.'''
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).repeat()

            normal_batch = self.dataset_batch_loader(dataset, self.train_high_normal, name='normal_loader')
            crop_batch = self.dataset_batch_loader(dataset, self.train_high_crop, name='crop_loader')
            brightness_batch = self.dataset_batch_loader(dataset, self.train_high_brightness, name='brightness_loader')
            contrast_batch = self.dataset_batch_loader(dataset, self.train_high_contrast, name='contrast_loader')
            hue_batch = self.dataset_batch_loader(dataset, self.train_high_hue, name='hue_loader')
            saturation_batch = self.dataset_batch_loader(dataset, self.train_high_saturation, name='saturation_loader')

        return normal_batch, crop_batch, brightness_batch, contrast_batch, hue_batch, saturation_batch

    def test_low_normal(self, x, y):
        with tf.variable_scope('test_low_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.LOW_IMG_SIZE)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def test_mid_normal(self, x, y):
        with tf.variable_scope('test_mid_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.MID_IMG_SIZE)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def test_high_normal(self, x, y):
        with tf.variable_scope('test_high_normal'):
            x = tf.read_file(x)
            x = tf.image.decode_png(x, channels=1, name='decode_img')
            x = tf.image.resize_images(x, size=DataLoader.HIGH_IMG_SIZE)
            x = self.tf_equalize_histogram(x)
            x = tf.divide(tf.cast(x, tf.float32), 255.)
        return x, y

    def test_low_loader(self):
        with tf.variable_scope('test_low_loader'):
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_x, self.test_y)).repeat()

            normal_dataset_map = dataset.map(self.test_low_normal).batch(self.batch_size)
            normal_iterator = normal_dataset_map.make_one_shot_iterator()
            normal_batch = normal_iterator.get_next()

        return normal_batch

    def test_mid_loader(self):
        with tf.variable_scope('test_mid_loader'):
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_x, self.test_y)).repeat()

            normal_dataset_map = dataset.map(self.test_mid_normal).batch(self.batch_size)
            normal_iterator = normal_dataset_map.make_one_shot_iterator()
            normal_batch = normal_iterator.get_next()

        return normal_batch

    def test_high_loader(self):
        with tf.variable_scope('test_high_loader'):
            dataset = tf.contrib.data.Dataset.from_tensor_slices((self.test_x, self.test_y)).repeat()

            normal_dataset_map = dataset.map(self.test_high_normal).batch(self.batch_size)
            normal_iterator = normal_dataset_map.make_one_shot_iterator()
            normal_batch = normal_iterator.get_next()

        return normal_batch