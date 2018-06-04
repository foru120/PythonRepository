import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import matplotlib.pyplot as plt

class TFRecordDataset:

    def __init__(self, tfrecord_dir, dataset_name, num_classes, split_name):
        self.tfrecord_dir = tfrecord_dir
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.split_name = split_name

    def _get_num_samples(self, split_name):
        num_samples = 0
        file_pattern_for_counting = self.dataset_name + '_' + split_name
        tfrecords_to_count = [os.path.join(self.tfrecord_dir, file) for file in os.listdir(self.tfrecord_dir) if
                              file.startswith(file_pattern_for_counting)]
        for tfrecord_file in tfrecords_to_count:
            for _ in tf.python_io.tf_record_iterator(tfrecord_file):
                num_samples += 1

        return num_samples

    def get_dataset(self, split_name):
        splits_to_sizes = self._get_num_samples(split_name)

        file_pattern = self.dataset_name + '_' + split_name + '_*.tfrecord'
        file_pattern = os.path.join(self.tfrecord_dir, file_pattern)

        reader = tf.TFRecordReader

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label')
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features=keys_to_features, items_to_handlers=items_to_handlers)

        return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=splits_to_sizes,
            items_to_descriptions='eye_right_train',
            num_classes=self.num_classes)

    def load_single(self, dataset):
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
        [image, label] = provider.get(['image', 'label'])

        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                plt.figure()
                for i in range(4):
                    train_img, train_label = sess.run([image, label])
                    height, width, _ = train_img.shape
                    plt.subplot(2, 2, i + 1)
                    plt.imshow(train_img)
                    plt.title('%s, %d x %d' % (train_label, height, width))
                    plt.axis('off')
                plt.show()

    def img_random_crop(self, image, size):
        with tf.variable_scope('img_random_crop'):
            low_img = tf.image.resize_images(images=image, size=(size[0] + 20, size[1] + 20),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            low_img = tf.random_crop(value=low_img, size=(size[0], size[1], 1))
            low_img = tf.cast(low_img, dtype=tf.float32)
        return low_img

    def data_normalize(self, data):
        return tf.divide(tf.cast(data, dtype=tf.float32), 255.)

    def load_batch(self, size, batch_size=10, num_classes=7, scope='load_batch'):
        with tf.variable_scope(name_or_scope=scope):
            dataset = self.get_dataset(split_name=self.split_name)
            # shuffle 옵션의 기본값이 True 이므로 data random sort 발생 (sort 가 발생하지 않도록 하기 위해 False 로 설정)
            provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=False)
            [image, label] = provider.get(['image', 'label'])
            image = tf.image.rgb_to_grayscale(image)

            "Random Crop Image 생성"
            random_crop_img = self.img_random_crop(image=image, size=size)

            "Normalization 수행"
            norm_img = self.data_normalize(random_crop_img)
            one_hot_label = slim.one_hot_encoding(label, num_classes)

            norm_imgs, one_hot_labels = tf.train.batch(
                [norm_img, one_hot_label],
                batch_size=batch_size,
                num_threads=1,
                capacity=2 * batch_size
            )

        return norm_imgs, one_hot_labels, dataset.num_samples

if __name__ == '__main__':
    RIGHT_TF_RECORD_DIR = 'G:\\04_dataset\\tfrecords\\eye\\train\\right'
    LEFT_TF_RECORD_DIR = 'G:\\04_dataset\\tfrecords\\eye\\train\\left'

    LOW_IMG_SIZE = (60, 160)
    MID_IMG_SIZE = (80, 200)
    HIGH_IMG_SIZE = (100, 240)

    BATCH_SIZE = 20

    tr_low_right_dataset = TFRecordDataset(tfrecord_dir=RIGHT_TF_RECORD_DIR, dataset_name='eye', num_classes=7, split_name='right_train')
    tr_mid_right_dataset = TFRecordDataset(tfrecord_dir=RIGHT_TF_RECORD_DIR, dataset_name='eye', num_classes=7, split_name='right_train')
    tr_high_right_dataset = TFRecordDataset(tfrecord_dir=RIGHT_TF_RECORD_DIR, dataset_name='eye', num_classes=7, split_name='right_train')

    tr_low_right_imgs, tr_low_right_labels, tot_num_samples = tr_low_right_dataset.load_batch(size=LOW_IMG_SIZE, batch_size=BATCH_SIZE, num_classes=7, scope='tr_low_right_batch')
    tr_mid_right_imgs, tr_mid_right_labels, _ = tr_mid_right_dataset.load_batch(size=MID_IMG_SIZE, batch_size=BATCH_SIZE, num_classes=7, scope='tr_mid_right_batch')
    tr_high_right_imgs, tr_high_right_labels, _ = tr_high_right_dataset.load_batch(size=HIGH_IMG_SIZE, batch_size=BATCH_SIZE, num_classes=7, scope='tr_high_right_batch')

    tr_low_left_dataset = TFRecordDataset(tfrecord_dir=LEFT_TF_RECORD_DIR, dataset_name='eye', num_classes=7, split_name='left_train')
    tr_mid_left_dataset = TFRecordDataset(tfrecord_dir=LEFT_TF_RECORD_DIR, dataset_name='eye', num_classes=7, split_name='left_train')
    tr_high_left_dataset = TFRecordDataset(tfrecord_dir=LEFT_TF_RECORD_DIR, dataset_name='eye', num_classes=7, split_name='left_train')

    tr_low_left_imgs, tr_low_left_labels, _ = tr_low_left_dataset.load_batch(size=LOW_IMG_SIZE, batch_size=BATCH_SIZE, num_classes=7, scope='tr_low_left_batch')
    tr_mid_left_imgs, tr_mid_left_labels, _ = tr_mid_left_dataset.load_batch(size=MID_IMG_SIZE, batch_size=BATCH_SIZE, num_classes=7, scope='tr_mid_left_batch')
    tr_high_left_imgs, tr_high_left_labels, _ = tr_high_left_dataset.load_batch(size=HIGH_IMG_SIZE, batch_size=BATCH_SIZE, num_classes=7, scope='tr_high_left_batch')

    step_number = tot_num_samples // BATCH_SIZE

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.3)
    )

    with tf.Session(config=config) as sess:
        with slim.queues.QueueRunners(sess):
            for step in range(step_number):
                print('>> ' + str(step))

                tr_low_right_img_batch, tr_low_right_label_batch, tr_mid_right_img_batch, tr_mid_right_label_batch, \
                tr_high_right_img_batch, tr_high_right_label_batch = sess.run([tr_low_right_imgs, tr_low_right_labels,
                                                                               tr_mid_right_imgs, tr_mid_right_labels,
                                                                               tr_high_right_imgs, tr_high_right_labels])

                tr_low_left_img_batch, tr_low_left_label_batch, tr_mid_left_img_batch, tr_mid_left_label_batch, \
                tr_high_left_img_batch, tr_high_left_label_batch = sess.run([tr_low_left_imgs, tr_low_left_labels,
                                                                             tr_mid_left_imgs, tr_mid_left_labels,
                                                                             tr_high_left_imgs, tr_high_left_labels])