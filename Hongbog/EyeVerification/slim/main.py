from Hongbog.EyeVerification.slim.model_v4_slim import Model
from Hongbog.EyeVerification.slim.dataloader import TFRecordDataset
from Hongbog.EyeVerification.slim.constants import *

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step

class NeuralNet:
    LOW_IMG_SIZE = (60, 160)
    MID_IMG_SIZE = (80, 200)
    HIGH_IMG_SIZE = (100, 240)

    def __init__(self, is_train, is_logging):
        if is_train:
            self._create_dataset()

    def _create_dataset(self):
        """Train Dataset Initialization"""
        tr_low_right_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_right_data_dir, dataset_name='eye', num_classes=7, split_name='right_train')
        tr_mid_right_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_right_data_dir, dataset_name='eye', num_classes=7, split_name='right_train')
        tr_high_right_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_right_data_dir, dataset_name='eye', num_classes=7, split_name='right_train')

        self.tr_low_right_imgs, self.tr_low_right_labels, self.tr_tot_nums = \
            tr_low_right_dataset.load_batch(size=NeuralNet.LOW_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='tr_low_right_batch')
        self.tr_mid_right_imgs, self.tr_mid_right_labels, _ = \
            tr_mid_right_dataset.load_batch(size=NeuralNet.MID_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='tr_mid_right_batch')
        self.tr_high_right_imgs, self.tr_high_right_labels, _ = \
            tr_high_right_dataset.load_batch(size=NeuralNet.HIGH_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='tr_high_right_batch')

        tr_low_left_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_left_data_dir, dataset_name='eye', num_classes=7, split_name='left_train')
        tr_mid_left_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_left_data_dir, dataset_name='eye', num_classes=7, split_name='left_train')
        tr_high_left_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_left_data_dir, dataset_name='eye', num_classes=7, split_name='left_train')

        self.tr_low_left_imgs, self.tr_low_left_labels, _ = \
            tr_low_left_dataset.load_batch(size=NeuralNet.LOW_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='tr_low_left_batch')
        self.tr_mid_left_imgs, self.tr_mid_left_labels, _ = \
            tr_mid_left_dataset.load_batch(size=NeuralNet.MID_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='tr_mid_left_batch')
        self.tr_high_left_imgs, self.tr_high_left_labels, _ = \
            tr_high_left_dataset.load_batch(size=NeuralNet.HIGH_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='tr_high_left_batch')
        self.tr_step_num = self.tr_tot_nums // flags.FLAGS.batch_size

        """Test Dataset Initialization"""
        ts_low_right_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_right_data_dir, dataset_name='eye', num_classes=7, split_name='right_test')
        ts_mid_right_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_right_data_dir, dataset_name='eye', num_classes=7, split_name='right_test')
        ts_high_right_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_right_data_dir, dataset_name='eye', num_classes=7, split_name='right_test')

        self.ts_low_right_imgs, self.ts_low_right_labels, self.ts_tot_nums = \
            ts_low_right_dataset.load_batch(size=NeuralNet.LOW_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='ts_low_right_batch')
        self.ts_mid_right_imgs, self.ts_mid_right_labels, _ = \
            ts_mid_right_dataset.load_batch(size=NeuralNet.MID_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='ts_mid_right_batch')
        self.ts_high_right_imgs, self.ts_high_right_labels, _ = \
            ts_high_right_dataset.load_batch(size=NeuralNet.HIGH_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='ts_high_right_batch')

        ts_low_left_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_left_data_dir, dataset_name='eye', num_classes=7, split_name='left_test')
        ts_mid_left_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_left_data_dir, dataset_name='eye', num_classes=7, split_name='left_test')
        ts_high_left_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_left_data_dir, dataset_name='eye', num_classes=7, split_name='left_test')

        self.ts_low_left_imgs, self.ts_low_left_labels, _ = \
            ts_low_left_dataset.load_batch(size=NeuralNet.LOW_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='ts_low_left_batch')
        self.ts_mid_left_imgs, self.ts_mid_left_labels, _ = \
            ts_mid_left_dataset.load_batch(size=NeuralNet.MID_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='ts_mid_left_batch')
        self.ts_high_left_imgs, self.ts_high_left_labels, _ = \
            ts_high_left_dataset.load_batch(size=NeuralNet.HIGH_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=7, scope='ts_high_left_batch')
        self.ts_step_num = self.ts_tot_nums // flags.FLAGS.batch_size

        print('>>> TFRecord Dataset Initialization Complete!')

    def train(self):
        tf.logging.set_verbosity(tf.logging.INFO)

        """Train Model Part"""
        right_model = Model(low_res_inputs=self.tr_low_right_imgs, mid_res_inputs=self.tr_mid_right_imgs,
                            high_res_inputs=self.tr_high_right_imgs, name='right_model')
        tr_right_logits = right_model.logits
        tr_right_loss = slim.losses.softmax_cross_entropy(logits=tr_right_logits, onehot_labels=self.tr_low_right_labels, scope='tr_right_ce_loss')
        tr_right_tot_loss = tf.add_n([tr_right_loss] +
                                     [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if 'right_model' in var.name], name='tr_right_tot_loss')

        right_opt = tf.train.GradientDescentOptimizer(learning_rate=flags.FLAGS.learning_rate, name='right_opt')

        right_train_op = slim.learning.create_train_op(
            total_loss=tr_right_tot_loss,
            optimizer=right_opt,
            variables_to_train= [var for var in tf.trainable_variables() if 'right_model' in var.name]
        )

        slim.learning.train(train_op=right_train_op,
                            logdir='D:\\Source\\PythonRepository\\Hongbog\\EyeVerification\\slim\\train_log\\test\\right',
                            number_of_steps=5,
                            save_summaries_secs=300,
                            save_interval_secs=1)

        left_model = Model(low_res_inputs=self.tr_low_left_imgs, mid_res_inputs=self.tr_mid_left_imgs,
                           high_res_inputs=self.tr_high_left_imgs, name='left_model')
        tr_left_logits = left_model.logits
        tr_left_loss = slim.losses.softmax_cross_entropy(logits=tr_left_logits, onehot_labels=self.tr_low_left_labels, scope='tr_left_ce_loss')
        tr_left_tot_loss = slim.losses.get_total_loss()

        left_opt = tf.train.GradientDescentOptimizer(learning_rate=flags.FLAGS.learning_rate, name='left_opt')

        left_train_op = slim.learning.create_train_op(
            total_loss=tr_left_tot_loss,
            optimizer=left_opt,
            variables_to_train=[var for var in tf.trainable_variables() if 'left_model' in var.name]
        )

        slim.learning.train(train_op=left_train_op,
                            logdir='D:\\Source\\PythonRepository\\Hongbog\\EyeVerification\\slim\\train_log\\test\\left',
                            number_of_steps=5,
                            save_summaries_secs=300,
                            save_interval_secs=1)

neural_net = NeuralNet(is_train=True, is_logging=True)
neural_net.train()