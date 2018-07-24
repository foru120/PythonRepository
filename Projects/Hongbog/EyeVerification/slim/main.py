from Hongbog.EyeVerification.slim.model_v4_slim import Model
from Hongbog.EyeVerification.slim.dataloader import TFRecordDataset
from Hongbog.EyeVerification.slim.constants import *

import tensorflow.contrib.slim as slim
import os

class Trainer(object):
    LOW_IMG_SIZE = (60, 160)
    MID_IMG_SIZE = (80, 200)
    HIGH_IMG_SIZE = (100, 240)

    def __init__(self, orientation, dataset_name, num_classes):
        self.orientation = orientation
        self.dataset_name = dataset_name
        self.num_classes = num_classes

        self.log_dir = 'D:\\Source\\PythonRepository\\Hongbog\\EyeVerification\\slim\\train_log\\test\\{}'.format(orientation)
        self.eval_dir = 'D:\\Source\\PythonRepository\\Hongbog\\EyeVerification\\slim\\train_log\\test_eval\\{}'.format(orientation)

    def _train_dataset(self):
        tr_low_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_right_data_dir if self.orientation == 'right' else flags.FLAGS.train_left_data_dir,
                                         dataset_name=self.dataset_name, num_classes=self.num_classes, split_name='{}_train'.format(self.orientation))
        tr_mid_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_right_data_dir if self.orientation == 'right' else flags.FLAGS.train_left_data_dir,
                                         dataset_name=self.dataset_name, num_classes=self.num_classes, split_name='{}_train'.format(self.orientation))
        tr_high_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.train_right_data_dir if self.orientation == 'right' else flags.FLAGS.train_left_data_dir,
                                          dataset_name=self.dataset_name, num_classes=self.num_classes, split_name='{}_train'.format(self.orientation))

        self.tr_low_imgs, self.tr_low_labels, self.tr_tot_nums = \
            tr_low_dataset.load_batch(size=Trainer.LOW_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=self.num_classes, scope='tr_low_batch')
        self.tr_mid_imgs, self.tr_mid_labels, _ = \
            tr_mid_dataset.load_batch(size=Trainer.MID_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=self.num_classes, scope='tr_mid_batch')
        self.tr_high_imgs, self.tr_high_labels, _ = \
            tr_high_dataset.load_batch(size=Trainer.HIGH_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=self.num_classes, scope='tr_high_batch')

        self.tr_step_num = self.tr_tot_nums // flags.FLAGS.batch_size
        print('>>> TFRecord Train Dataset Initialization Complete!')

    def _validation_dataset(self):
        ts_low_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_right_data_dir if self.orientation == 'right' else flags.FLAGS.test_left_data_dir,
                                         dataset_name=self.dataset_name, num_classes=self.num_classes, split_name='{}_test'.format(self.orientation))
        ts_mid_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_right_data_dir if self.orientation == 'right' else flags.FLAGS.test_left_data_dir,
                                         dataset_name=self.dataset_name, num_classes=self.num_classes, split_name='{}_test'.format(self.orientation))
        ts_high_dataset = TFRecordDataset(tfrecord_dir=flags.FLAGS.test_right_data_dir if self.orientation == 'right' else flags.FLAGS.test_left_data_dir,
                                          dataset_name=self.dataset_name, num_classes=self.num_classes, split_name='{}_test'.format(self.orientation))

        self.ts_low_imgs, self.ts_low_labels, self.ts_tot_nums = \
            ts_low_dataset.load_batch(size=Trainer.LOW_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=self.num_classes, scope='ts_low_batch')
        self.ts_mid_imgs, self.ts_mid_labels, _ = \
            ts_mid_dataset.load_batch(size=Trainer.MID_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=self.num_classes, scope='ts_mid_batch')
        self.ts_high_imgs, self.ts_high_labels, _ = \
            ts_high_dataset.load_batch(size=Trainer.HIGH_IMG_SIZE, batch_size=flags.FLAGS.batch_size, num_classes=self.num_classes, scope='ts_high_batch')

        self.ts_step_num = self.ts_tot_nums // flags.FLAGS.batch_size
        print('>>> TFRecord Validation Dataset Initialization Complete!')

    def train(self, epoch):
        with tf.variable_scope(name_or_scope=self.orientation):
            tf.logging.set_verbosity(tf.logging.INFO)

            """Train Model Part"""
            self._train_dataset()

            tr_model = Model(low_res_inputs=self.tr_low_imgs, mid_res_inputs=self.tr_mid_imgs, high_res_inputs=self.tr_high_imgs)
            tr_logits = tr_model.logits
            tr_loss = slim.losses.softmax_cross_entropy(logits=tr_logits, onehot_labels=self.tr_low_labels, scope='tr_ce_loss')
            tr_tot_loss = tf.add_n([tr_loss] +
                                   [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.orientation in var.name], name='tr_tot_loss')

            opt = tf.train.GradientDescentOptimizer(learning_rate=flags.FLAGS.learning_rate, name='opt')

            """텐서보드 시각화를 위한 훈련 메트릭 정의"""
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(tr_logits, 1), tf.arg_max(self.tr_low_labels, 1)), tf.float32))
            tf.summary.scalar('losses/Total', tr_tot_loss)
            tf.summary.scalar('accuracy', accuracy)
            summary_op = tf.summary.merge_all()

            """훈련 수행"""
            os.makedirs(self.log_dir, exist_ok=True)

            # update_ops = [] : 모든 update_ops 를 사용하지 않도록 설정
            train_op = slim.learning.create_train_op(
                total_loss=tr_tot_loss,
                optimizer=opt,
                update_ops=[var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.orientation in var.name],
                variables_to_train= [var for var in tf.trainable_variables() if self.orientation in var.name]
            )

            saver = tf.train.Saver(tf.global_variables())

            slim.learning.train(
                train_op=train_op,
                logdir=self.log_dir,
                number_of_steps=self.tr_step_num * epoch,
                summary_op=summary_op,
                save_summaries_secs=60,
                save_interval_secs=60,
                saver=saver
            )

    def validation(self):
        with tf.variable_scope(name_or_scope=self.orientation):
            tf.logging.set_verbosity(tf.logging.INFO)

            """Validation Model Part"""
            self._validation_dataset()

            ts_model = Model(low_res_inputs=self.ts_low_imgs, mid_res_inputs=self.ts_mid_imgs, high_res_inputs=self.ts_high_imgs, is_training=False)
            ts_logit = ts_model.logits

            "평가 메트릭 정의"
            names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                'eval/Accuracy': slim.metrics.streaming_accuracy(tf.arg_max(ts_logit, 1), tf.arg_max(self.ts_low_labels, 1))
            })

            """평가 수행"""
            # init_op = tf.group(
            #     tf.global_variables_initializer(),
            #     tf.local_variables_initializer()
            # )

            checkpoint_path = tf.train.latest_checkpoint(self.log_dir)

            os.makedirs(self.eval_dir, exist_ok=True)

            print([var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            print([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if ('moving_mean' in var.name)
                          or ('moving_variance' in var.name)])

            restore_variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)] + \
                                [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                                 ('moving_mean' in var.name)]

            slim.get_or_create_global_step()
            metric_values = slim.evaluation.evaluate_once(
                master='',
                checkpoint_path=checkpoint_path,
                logdir=self.eval_dir,
                num_evals=self.ts_step_num,
                # initial_op=init_op,
                eval_op=list(names_to_updates.values()),  # names_to_updates.values() -> type 이 dict_values 이어서 list 로 변환시켜주어야 한다.
                final_op=list(names_to_values.values()),
                variables_to_restore=tf.global_variables()
            )

            print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'right/high_resolution_network/residual_block_18/batch_norm_2/moving_variance:0')[0])

            names_to_values = dict(zip(names_to_values.keys(), metric_values))
            for name in names_to_values:
                print('%s: %f' % (name, names_to_values[name]))

trainer = Trainer(orientation='right', dataset_name='eye', num_classes=7)
# trainer.validation()
trainer.train(epoch=30)