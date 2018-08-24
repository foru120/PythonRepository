import os
import re
import math
import numpy as np
from Projects.Hongbog.EyeVerification.slim.tfrecord_features import *

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
    def __init__(self, dataset_name, right_dataset_dir, left_dataset_dir, output_dataset_dir, sample_per_file, test_ratio, shuffling=True):
        self.dataset_name = dataset_name
        self.right_dataset_dir = right_dataset_dir
        self.left_dataset_dir = left_dataset_dir
        self.output_dataset_dir = output_dataset_dir
        self.sample_per_file = sample_per_file
        self.test_ratio = test_ratio
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

    def _write_to_tfrecord(self, category, direction, fnames, labels, tot_cnt):
        fname_cnt = 0
        tfrecord_fname = 1
        max_tfrecord_fname = math.ceil(tot_cnt / self.sample_per_file)
        eof = False

        with tf.Graph().as_default():
            image_reader = ImageReader()

            with tf.Session() as sess:
                while True:
                    os.makedirs(os.path.join(self.output_dataset_dir, self.dataset_name, category, direction), exist_ok=True)
                    cur_tfrecord_fname = os.path.join(self.output_dataset_dir, self.dataset_name, category, direction,
                                                      self.dataset_name + '_' + direction + '_' + category + '_' + str(tfrecord_fname).zfill(5)
                                                      + '-of-' + str(max_tfrecord_fname).zfill(5) + '.tfrecord')
                    print('>> ' + cur_tfrecord_fname + ', file create.')

                    with tf.python_io.TFRecordWriter(cur_tfrecord_fname) as tfrecord_writer:
                        for _ in range(self.sample_per_file):
                            if fname_cnt == tot_cnt:
                                eof = True
                                break

                            self._add_to_tfrecord(fnames[fname_cnt], labels[fname_cnt], tfrecord_writer, image_reader, sess)
                            fname_cnt += 1
                        tfrecord_fname += 1

                    if eof:
                        break

    def run(self):
        "파일 디렉토리 존재 유/무 체크"
        if not tf.gfile.Exists(self.right_dataset_dir):
            raise Exception('>> Not Exists {} Directory.'.format(self.right_dataset_dir))

        if not tf.gfile.Exists(self.left_dataset_dir):
            raise Exception('>> Not Exists {} Directory.'.format(self.left_dataset_dir))

        "디렉토리에 해당하는 파일 명/라벨 배열 화"
        tot_right_fnames, tot_right_labels = [], []
        tot_left_fnames, tot_left_labels = [], []
        label_reg = re.compile('.*\\\\(\d)+')

        for (path, dirs, files) in os.walk(self.right_dataset_dir):
            for file in files:
                tot_right_fnames.append(os.path.join(path, file))
                tot_right_labels.append(label_reg.match(path).group(1))

        for (path, dirs, files) in os.walk(self.left_dataset_dir):
            for file in files:
                tot_left_fnames.append(os.path.join(path, file))
                tot_left_labels.append(label_reg.match(path).group(1))

        tot_right_fnames = np.asarray(tot_right_fnames)
        tot_right_labels = np.asarray(tot_right_labels, dtype=np.int64)
        tot_left_fnames = np.asarray(tot_left_fnames)
        tot_left_labels = np.asarray(tot_left_labels, dtype=np.int64)

        "랜덤 정렬 수행"
        tot_cnt = tot_right_fnames.shape[0]  # 전체 파일 개수

        if self.shuffling:
            rnd_sort = np.random.permutation(tot_cnt)
            tot_right_fnames = tot_right_fnames[rnd_sort]
            tot_right_labels = tot_right_labels[rnd_sort]
            tot_left_fnames = tot_left_fnames[rnd_sort]
            tot_left_labels = tot_left_labels[rnd_sort]

        "Train/Validation 데이터 셋으로 분할"
        train_right_fnames = tot_right_fnames[:int(tot_cnt * (1 - self.test_ratio))]
        train_right_labels = tot_right_labels[:int(tot_cnt * (1 - self.test_ratio))]
        train_left_fnames = tot_left_fnames[:int(tot_cnt * (1 - self.test_ratio))]
        train_left_labels = tot_left_labels[:int(tot_cnt * (1 - self.test_ratio))]

        test_right_fnames = tot_right_fnames[int(tot_cnt * (1 - self.test_ratio)):]
        test_right_labels = tot_right_labels[int(tot_cnt * (1 - self.test_ratio)):]
        test_left_fnames = tot_left_fnames[int(tot_cnt * (1 - self.test_ratio)):]
        test_left_labels = tot_left_labels[int(tot_cnt * (1 - self.test_ratio)):]

        "TFRecord 데이터 셋 생성"
        self._write_to_tfrecord(category='train', direction='right', fnames=train_right_fnames, labels=train_right_labels, tot_cnt=train_right_fnames.shape[0])
        self._write_to_tfrecord(category='train', direction='left', fnames=train_left_fnames, labels=train_left_labels, tot_cnt=train_left_fnames.shape[0])
        self._write_to_tfrecord(category='test', direction='right', fnames=test_right_fnames, labels=test_right_labels, tot_cnt=test_right_fnames.shape[0])
        self._write_to_tfrecord(category='test', direction='left', fnames=test_left_fnames, labels=test_left_labels, tot_cnt=test_left_fnames.shape[0])

if __name__ == '__main__':
    tfrecord = TFRecord(dataset_name='eye',
                        right_dataset_dir='E:\\04_dataset\\eye_verification\\eye_only_v6\\right',
                        left_dataset_dir='E:\\04_dataset\\eye_verification\\eye_only_v6\\left',
                        output_dataset_dir='E:\\04_dataset\\tfrecords',
                        sample_per_file=1000, test_ratio=0.3)
    tfrecord.run()
