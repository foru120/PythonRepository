import os
import numpy as np
import tensorflow as tf

class Augmentation:
    IMG_SIZE = (220, 400)

    def __init__(self, asis_dir, tobe_dir, sort, sample_num):
        self.asis_dir = asis_dir
        self.tobe_dir = tobe_dir
        self.sort = sort
        self.sample_num = sample_num
        self.batch_size = 1
        self.ref_func = {'random_crop': self.random_crop,
                         'random_brightness': self.random_brightness,
                         'random_contrast': self.random_contrast,
                         'random_hue': self.random_hue,
                         'random_saturation': self.random_saturation}

    def data_setting(self):
        self.data = []

        for (path, dirs, files) in os.walk(self.asis_dir):
            for file in files:
                self.data.append(os.path.join(path, file))

        self.data_num = len(self.data)

        with tf.variable_scope('image_data'):
            self.data = tf.convert_to_tensor(self.data, dtype=tf.string, name='data')

    def create_img(self):
        loader = self.data_loader()

        with tf.Session() as sess:
            for idx in range(self.sample_num):
                for l, s in zip(loader, self.sort):
                    with open(os.path.join(self.tobe_dir, s + '_' + str(idx) + '.jpg'), mode='wb') as f:
                        f.write(sess.run(tf.image.encode_jpeg(tf.squeeze(l, [0]))))

    def dataset_setting(self, dataset, ref_func, name):
        with tf.variable_scope(name_or_scope=name):
            dataset_map = dataset.map(ref_func).batch(self.batch_size)
            iterator = dataset_map.make_one_shot_iterator()
            batch = iterator.get_next()
        return batch

    def data_loader(self):
        loader_list = []

        with tf.variable_scope('data_loader'):
            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.data).repeat()

            for s in self.sort:
                loader_list.append(self.dataset_setting(dataset=dataset, ref_func=self.ref_func[s], name=s))

        return loader_list

    def random_crop(self, x):
        with tf.variable_scope('random_crop'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=(240, 440))
            x = tf.random_crop(value=x, size=(Augmentation.IMG_SIZE[0], Augmentation.IMG_SIZE[1], 3))
            x = tf.cast(x, tf.uint8)
        return x

    def random_brightness(self, x):
        with tf.variable_scope('random_brightness'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=Augmentation.IMG_SIZE)
            x = tf.image.random_brightness(x, max_delta=80.)
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.image.random_saturation(x, lower=0.2, upper=2.0)
            x = tf.clip_by_value(x, 0.0, 255.0)
            x = tf.cast(x, tf.uint8)
        return x

    def random_contrast(self, x):
        with tf.variable_scope('random_contrast'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=Augmentation.IMG_SIZE)
            x = tf.image.random_contrast(x, lower=0.2, upper=2.0)
            x = tf.cast(x, tf.uint8)
        return x

    def random_hue(self, x):
        with tf.variable_scope('random_hue'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=Augmentation.IMG_SIZE)
            x = tf.image.random_hue(x, max_delta=0.08)
            x = tf.cast(x, tf.uint8)
        return x

    def random_saturation(self, x):
        with tf.variable_scope('random_saturation'):
            x = tf.read_file(x)
            x = tf.image.decode_jpeg(x, channels=3, name='decode_img')
            x = tf.image.resize_images(x, size=Augmentation.IMG_SIZE)
            x = tf.image.random_saturation(x, lower=0.2, upper=2.0)
            x = tf.cast(x, tf.uint8)
        return x

augmentation = Augmentation(asis_dir='/home/kyh/PycharmProjects/PythonRepository/Projects/DeepLearningTechniques/Augmentation/asis_img',
                            tobe_dir='/home/kyh/PycharmProjects/PythonRepository/Projects/DeepLearningTechniques/Augmentation/tobe_img',
                            sort=['random_brightness'],
                            sample_num=5)
augmentation.data_setting()
augmentation.create_img()