import os
import re
import math
import numpy as np
from Projects.Hongbog.MultiGPU.tfrecord_converting.tfrecord_features import *

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        return image

class TFRecord:
    def __init__(self, dataset_name, dataset_dir, output_dataset_dir, sample_per_file, shuffling=True):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.output_dataset_dir = output_dataset_dir
        self.sample_per_file = sample_per_file
        self.shuffling = shuffling

    def _add_to_tfrecord(self, fname, label, tfrecord_writer, image_reader, sess):
        img_data = tf.gfile.FastGFile(fname, 'rb').read()
        # height, width = image_reader.read_image_dims(sess, img_data)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(img_data),
            'image/format': bytes_feature(b'png'),
            'image/class/label': int64_feature(label)
        }))

        tfrecord_writer.write(example.SerializeToString())

    def _write_to_tfrecord(self, category, x, y):
        tot_data_cnt = x.shape[0]
        data_cnt = 0
        tfrecord_fname = 1
        max_tfrecord_fname = tot_data_cnt // self.sample_per_file
        eof = False

        with tf.Graph().as_default():
            image_reader = ImageReader()

            with tf.Session() as sess:
                while True:
                    os.makedirs(os.path.join(self.output_dataset_dir, self.dataset_name, category), exist_ok=True)
                    cur_tfrecord_fname = os.path.join(self.output_dataset_dir, self.dataset_name, category,
                                                      self.dataset_name + '_' + category + '_' + str(tfrecord_fname).zfill(5)
                                                      + '-of-' + str(max_tfrecord_fname).zfill(5) + '.tfrecord')
                    print('>> ' + cur_tfrecord_fname + ', file create.')

                    with tf.python_io.TFRecordWriter(cur_tfrecord_fname) as tfrecord_writer:
                        for _ in range(self.sample_per_file):
                            if data_cnt == tot_data_cnt - 1:
                                eof = True
                                break

                            self._add_to_tfrecord(x[data_cnt], y[data_cnt], tfrecord_writer, image_reader, sess)
                            data_cnt += 1
                        tfrecord_fname += 1

                    if eof:
                        break

    def run(self):
        #todo 파일 디렉토리 존재 유/무 체크
        if not tf.gfile.Exists(self.dataset_dir):
            raise Exception('>> Not Exists {} Directory.'.format(self.dataset_dir))

        #todo 디렉토리에 해당하는 파일 명/라벨 배열 화
        train_x, train_y = [], []
        validation_x, validation_y = [], []
        test_x, test_y = [], []

        division = {'train': [train_x, train_y],
                    'validation': [validation_x, validation_y],
                    'test': [test_x, test_y]}

        label_reg = re.compile('.*\\\(\d)+')

        for key in division.keys():
            for (path, dirs, files) in os.walk(os.path.join(self.dataset_dir, key)):
                for file in files:
                    division[key][0].append(os.path.join(path, file))
                    division[key][1].append(label_reg.match(path).group(1))

        train_x, train_y = np.asarray(train_x), np.asarray(train_y, dtype=np.int64)
        validation_x, validation_y = np.asarray(validation_x), np.asarray(validation_y, dtype=np.int64)
        test_x, test_y = np.asarray(test_x), np.asarray(test_y, dtype=np.int64)

        if self.shuffling:
            train_rand_sort = np.random.permutation(len(train_x))
            validation_rand_sort = np.random.permutation(len(validation_x))
            test_rand_sort = np.random.permutation(len(test_x))

            train_x, train_y = train_x[train_rand_sort], train_y[train_rand_sort]
            validation_x, validation_y = validation_x[validation_rand_sort], validation_y[validation_rand_sort]
            test_x, test_y = test_x[test_rand_sort], test_y[test_rand_sort]

        #todo TFRecord 데이터 셋 생성
        if train_x.size > 0: self._write_to_tfrecord(category='train', x=train_x, y=train_y)
        if validation_x.size > 0: self._write_to_tfrecord(category='validation', x=validation_x, y=validation_y)
        if test_x.size > 0: self._write_to_tfrecord(category='test', x=test_x, y=test_y)

if __name__ == '__main__':
    tfrecord = TFRecord(dataset_name='cifar10',
                        dataset_dir='G:/04.dataset/02.cifar10/cifar10_original',
                        output_dataset_dir='G:/04.dataset/02.cifar10/cifar10_tfrecord_bak',
                        sample_per_file=1000,
                        shuffling=True)
    tfrecord.run()