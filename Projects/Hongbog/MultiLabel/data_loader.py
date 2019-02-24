import os
import re

from sklearn.preprocessing import MultiLabelBinarizer

from Projects.Hongbog.MultiLabel.constants import *

class DataLoader:
    def __init__(self):
        self.data_root_path = flags.FLAGS.data_root_path

    def read_images_from_disk(self, queue):
        label = queue[1]
        image = tf.read_file(queue[0])
        image = tf.image.decode_jpeg(image, channels=1)
        image.set_shape([flags.FLAGS.image_height, flags.FLAGS.image_width, flags.FLAGS.image_channel])
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)

        return image, label

    def train_batch(self):
        file_names = []
        file_labels = []
        label_reg = re.compile('\d+((\_\d){1,4}).jpg')

        for (path, dirs, files) in os.walk(os.path.join(self.data_root_path, 'train')):
            for file in files:
                full_name = os.path.join(path, file)
                multi_label = label_reg.match(file).group(1).lstrip('_').split('_')
                label_idx = [int(idx) for idx in multi_label]

                file_names.append(full_name)
                file_labels.append(label_idx)

        label_encoder = MultiLabelBinarizer()
        file_labels = label_encoder.fit_transform(file_labels)

        x = tf.convert_to_tensor(file_names, dtype=tf.string)
        y = tf.convert_to_tensor(file_labels, dtype=tf.float32)

        input_queue = tf.train.slice_input_producer([x, y], num_epochs=None, shuffle=True)

        x_queue, y_queue = self.read_images_from_disk(input_queue)

        train_x, train_y = tf.train.batch([x_queue, y_queue], batch_size=flags.FLAGS.batch_size)

        return train_x, train_y

    def test_batch(self):
        file_names = []
        file_labels = []
        label_reg = re.compile('\d+((\_\d){1,4}).jpg')

        for (path, dirs, files) in os.walk(os.path.join(self.data_root_path, 'test')):
            for file in files:
                full_name = os.path.join(path, file)
                multi_label = label_reg.match(file).group(1).lstrip('_').split('_')
                label_idx = [int(idx) for idx in multi_label]

                file_names.append(full_name)
                file_labels.append(label_idx)

        label_encoder = MultiLabelBinarizer()
        file_labels = label_encoder.fit_transform(file_labels)

        x = tf.convert_to_tensor(file_names, dtype=tf.string)
        y = tf.convert_to_tensor(file_labels, dtype=tf.float32)

        input_queue = tf.train.slice_input_producer([x, y], num_epochs=None, shuffle=True)

        x_queue, y_queue = self.read_images_from_disk(input_queue)

        test_x, test_y = tf.train.batch([x_queue, y_queue], batch_size=flags.FLAGS.batch_size)

        return test_x, test_y

if __name__ == '__main__':
    dataloader = DataLoader()
    dataloader.train_batch()