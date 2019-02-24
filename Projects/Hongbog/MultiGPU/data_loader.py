import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow.contrib.slim as slim

from Projects.Hongbog.MultiGPU.constants import *

class DataLoader:
    # todo train/test => (클래스 당 5000/1000)

    def __init__(self):
        self.image_width = flags.FLAGS.image_width
        self.image_height = flags.FLAGS.image_height
        self.batch_size = flags.FLAGS.batch_size
        self.root_data_path = flags.FLAGS.root_data_path
        self.dataset_name = flags.FLAGS.dataset_name
        self.num_classes = flags.FLAGS.num_classes

    def train_batch(self):
        with tf.variable_scope(name_or_scope='train_data_loader'):
            #todo TFRecord Dataset Initialization
            train_data_path = os.path.join(self.root_data_path, 'train')
            dataset = self._get_dataset(data_path=train_data_path)

            provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=True)
            [image, label] = provider.get(['image', 'label'])

            #todo Image Augmentation & Normalization
            image = self.distorted_image(image)

            train_x, train_y = tf.train.batch(
                [image, label],
                batch_size=self.batch_size,
                num_threads=16,
                capacity=5 * self.batch_size
            )

        return train_x, train_y

    def test_batch(self):
        with tf.variable_scope(name_or_scope='test_data_loader'):
            #todo TFRecord Dataset Initialization
            test_data_path = os.path.join(self.root_data_path, 'test')
            dataset = self._get_dataset(data_path=test_data_path, type='test')

            provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=True)
            [image, label] = provider.get(['image', 'label'])

            #todo Image Augmentation & Normalization
            image = self.normal_image(image)

            test_x, test_y = tf.train.batch(
                [image, label],
                batch_size=self.batch_size,
                num_threads=8,
                capacity=5 * self.batch_size
            )

        return test_x, test_y

    def _get_num_samples(self, data_path):
        num_samples = 0

        tfrecords_to_count = [os.path.join(data_path, file) for file in os.listdir(data_path)]
        for tfrecord_file in tfrecords_to_count:
            for _ in tf.python_io.tf_record_iterator(tfrecord_file):
                num_samples += 1

        return num_samples

    def _get_dataset(self, data_path, type='train'):
        num_samples = self._get_num_samples(data_path=data_path)

        file_pattens = self.dataset_name + '_' + type + '_*.tfrecord'
        data_sources = os.path.join(data_path, file_pattens)

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'label': slim.tfexample_decoder.Tensor('image/class/label')
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features=keys_to_features,
            items_to_handlers=items_to_handlers
        )

        return slim.dataset.Dataset(
            data_sources=data_sources,
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions='cifar10',
            num_classes=self.num_classes
        )

    def distorted_image(self, image):
        with tf.variable_scope(name_or_scope='distorted_image'):
            image = tf.cast(image, tf.float32)
            # image = tf.image.resize_images(images=image, size=(self.image_height + 5, self.image_width + 5), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=self.image_height + 4, target_width=self.image_width + 4)
            # image = tf.image.resize_image_with_pad(image=image, target_height=self.image_height + 4, target_width=self.image_width + 4)
            image = tf.random_crop(value=image, size=(self.image_height, self.image_width, 3))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=63)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            image = tf.divide(image, 255.)
            image = tf.subtract(image, [0.4914, 0.4822, 0.4465])
            image = tf.divide(image, [0.2470, 0.2435, 0.2616])
            # image = tf.image.per_image_standardization(image)

        return image

    def normal_image(self, image):
        with tf.variable_scope(name_or_scope='normal_image'):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize_images(images=image, size=(self.image_height, self.image_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            image = tf.divide(image, 255.)
            image = tf.subtract(image, [0.4914, 0.4822, 0.4465])
            image = tf.divide(image, [0.2470, 0.2435, 0.2616])
            # image = tf.image.per_image_standardization(image)

        return image

if __name__ == '__main__':
    loader = DataLoader()

    train_x, train_y = loader.train_batch()

    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [train_x, train_y], capacity=2 * flags.FLAGS.num_gpus
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(10):
            x, y = batch_queue.dequeue()
            batch_x, batch_y = sess.run([x, y])

            print(batch_x[0])
            print(batch_y[0])
            plt.imshow(batch_x[0])
            plt.show()

        coord.request_stop()
        coord.join(threads)

