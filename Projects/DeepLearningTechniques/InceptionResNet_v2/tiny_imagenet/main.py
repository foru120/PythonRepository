import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

from Projects.DeepLearningTechniques.InceptionResNet_v2.tiny_imagenet.constants import *
from Projects.DeepLearningTechniques.InceptionResNet_v2.tiny_imagenet.data_loader import DataLoader
from Projects.DeepLearningTechniques.InceptionResNet_v2.tiny_imagenet.inception_resnet_v2_model import Model
from Projects.DeepLearningTechniques.InceptionResNet_v2.tiny_imagenet.database import Database
from Projects.DeepLearningTechniques.InceptionResNet_v2.tiny_imagenet.cam import GradCAM

class Trainer:

    def __init__(self, seq, is_db_logging, is_ckpt_logging, is_cfm_logging, is_roc_logging, is_tb_logging, dataset, name):
        self.seq = seq
        self.is_db_logging = is_db_logging
        self.is_ckpt_logging = is_ckpt_logging
        self.is_cfm_logging = is_cfm_logging
        self.is_roc_logging = is_roc_logging
        self.is_tb_logging = is_tb_logging
        self.dataset = dataset
        self.name = name

        self.db = Database()
        print('>> Database connection is complete.')

        self._loader = DataLoader()
        print('>> The data loader has been initialized.')

    def train(self):
        self._loader.init_train()
        self._loader.init_validation()

        train_step_num = self._loader.train_len // flags.FLAGS.batch_size
        valid_step_num = self._loader.valid_len // flags.FLAGS.batch_size

        train_normal_batch, train_random_crop_batch = self._loader.train_loader()
        valid_normal_batch = self._loader.valid_loader()

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True,  # GPU 메모리 증분 할당
                                      per_process_gpu_memory_fraction=0.8)  # GPU 당 할당할 메모리 양
        )

        with tf.Session(config=config) as sess:
            #todo Model 객체 생성
            train_model = Model(sess=sess, is_training=True, is_tb_logging=self.is_tb_logging, name=self.name)

            valid_model = Model(sess=sess, is_training=False, is_tb_logging=self.is_tb_logging, name=self.name)

            sess.run(tf.global_variables_initializer())

            print('>> Tensorflow session built. Variables initialized.')

            #todo 훈련 시 필요한 로그 디렉토리 생성
            os.makedirs(os.path.join(flags.FLAGS.train_log_path, str(self.seq)), exist_ok=True)
            os.makedirs(os.path.join(flags.FLAGS.deploy_log_path, str(self.seq)), exist_ok=True)
            os.makedirs(os.path.join(flags.FLAGS.cam_log_path, str(self.seq)), exist_ok=True)
            os.makedirs(os.path.join(flags.FLAGS.tensorboard_log_path, str(self.seq)), exist_ok=True)
            os.makedirs(os.path.join(flags.FLAGS.roc_curve_log_path, str(self.seq)), exist_ok=True)
            os.makedirs(os.path.join(flags.FLAGS.cfm_log_path, str(self.seq)), exist_ok=True)

            #todo 텐서플로우 그래프 저장
            tf.train.write_graph(sess.graph_def, os.path.join(flags.FLAGS.train_log_path, str(self.seq)), 'train_graph.pbtxt')

            print('>> Tensorflow graph saved.')

            #todo Saver 생성
            self._train_saver = tf.train.Saver(var_list=tf.global_variables(self.name), max_to_keep=flags.FLAGS.epoch)

            #todo 텐서보드 로깅을 위한 FileWriter 생성
            if self.is_tb_logging:
                tensorboard_writer = tf.summary.FileWriter(os.path.join(flags.FLAGS.tensorboard_log_path, str(self.seq)), graph=tf.get_default_graph())

            best_valid_acc, best_valid_loss = 0., math.inf
            earlystop_patience, decay_patience = 0, 0

            #todo Total Train/validation
            print('>> Start learning.')
            tot_train_st = time.time()
            for epoch in range(1, flags.FLAGS.epoch + 1):
                train_acc_per_epoch, train_loss_per_epoch = [], []
                valid_acc_per_epoch, valid_loss_per_epoch = [], []
                valid_prob_per_epoch, valid_label_per_epoch = [], []

                #todo Train per epoch
                print('>> Epoch [%d]' % (epoch))
                epoch_train_st = time.time()
                for step in range(1, train_step_num + 1):
                    train_data = sess.run([train_normal_batch, train_random_crop_batch])
                    train_batch_x, train_batch_y = np.concatenate([data[0] for data in train_data]), np.concatenate([data[1] for data in train_data])

                    train_acc_per_step, train_loss_per_step = [], []
                    step_train_st = time.time()
                    for idx in range(0, flags.FLAGS.batch_size * 2, flags.FLAGS.batch_size):  # 2: augmentation 개수
                        if self.is_tb_logging:
                            train_acc, train_loss, train_summary, _ = \
                                train_model.train(x=train_batch_x[idx: idx + flags.FLAGS.batch_size],
                                                  y=train_batch_y[idx: idx + flags.FLAGS.batch_size])
                        else:
                            train_acc, train_loss, _ = \
                                train_model.train(x=train_batch_x[idx: idx + flags.FLAGS.batch_size],
                                                  y=train_batch_y[idx: idx + flags.FLAGS.batch_size])

                        train_acc_per_step.append(train_acc)
                        train_loss_per_step.append(train_loss)
                        train_acc_per_epoch.append(train_acc)
                        train_loss_per_epoch.append(train_loss)
                    step_train_et = time.time()

                    train_acc_per_step = np.sum(train_acc_per_step) / len(train_acc_per_step)
                    train_loss_per_step = np.sum(train_loss_per_step) / len(train_loss_per_step)

                    print('>> [Step-Train] epoch/step [%d/%d], acc: %.6f, loss: %.6f, time: %.2f'
                          % (epoch, step, train_acc_per_step, train_loss_per_step, (step_train_et - step_train_st)))
                epoch_train_et = time.time()

                # todo Tensorboard logging
                if self.is_tb_logging:
                    tensorboard_writer.add_summary(summary=train_summary, global_step=epoch)

                # todo Checkpoint file logging
                if self.is_ckpt_logging:
                    self._train_saver.save(sess, os.path.join(flags.FLAGS.train_log_path, str(self.seq), 'train_tiny_imagenet'), global_step=epoch)

                # todo Validation per epoch
                epoch_valid_st = time.time()
                for step in range(1, valid_step_num + 1):
                    valid_data = sess.run(valid_normal_batch)
                    valid_batch_x, valid_batch_y = valid_data[0], valid_data[1]

                    valid_acc, valid_loss, valid_prob = valid_model.validation(valid_batch_x, valid_batch_y)

                    valid_acc_per_epoch.append(valid_acc)
                    valid_loss_per_epoch.append(valid_loss)
                    valid_prob_per_epoch.append(valid_prob)
                    valid_label_per_epoch.append(valid_batch_y)
                epoch_valid_et = time.time()

                train_acc_per_epoch = np.sum(train_acc_per_epoch) / len(train_acc_per_epoch)
                train_loss_per_epoch = np.sum(train_loss_per_epoch) / len(train_loss_per_epoch)
                valid_acc_per_epoch = np.sum(valid_acc_per_epoch) / len(valid_acc_per_epoch)
                valid_loss_per_epoch = np.sum(valid_loss_per_epoch) / len(valid_loss_per_epoch)

                print('>> [Epoch-Train] epoch: [%d], acc: %.6f, loss: %.6f, time: %.2f'
                      % (epoch, train_acc_per_epoch, train_loss_per_epoch, (epoch_train_et - epoch_train_st)))
                print('>> [Epoch-Validation] epoch: [%d], acc: %.6f, loss: %.6f, time: %.2f'
                      % (epoch, valid_acc_per_epoch, valid_loss_per_epoch, (epoch_valid_et - epoch_valid_st)))

                #todo Database logging
                if self.is_db_logging:
                    self.db.mon_data_to_db(self.dataset, self.name, 'train', self.seq, epoch, train_acc_per_epoch, train_loss_per_epoch, (epoch_train_et - epoch_train_st))
                    self.db.mon_data_to_db(self.dataset, self.name, 'validation', self.seq, epoch, valid_acc_per_epoch, valid_loss_per_epoch, (epoch_valid_et - epoch_valid_st))

                #todo Confusion matrix
                print('>> [Confusion-Matrix] Epoch %d -' % (epoch))
                if self.is_cfm_logging:
                    cfm_str = ''
                    for line in sess.run(tf.confusion_matrix(labels=np.asarray(valid_label_per_epoch).flatten(),
                                                             predictions=np.argmax(np.asarray(valid_prob_per_epoch), axis=-1).flatten(),
                                                             num_classes=flags.FLAGS.image_class)):
                        temp = ''
                        for element in line:
                            fill_str = str(element).replace(' ', '').zfill(4)
                            for idx, ch in enumerate(fill_str):
                                if ch == '0' and idx < 3:
                                    temp += ' '
                                else:
                                    temp += fill_str[idx:]
                                    break
                            temp += ' '
                        cfm_str = cfm_str + temp + '\n'

                    with open(os.path.join(flags.FLAGS.cfm_log_path, str(self.seq), str(epoch) + '.txt'), mode='w') as f:
                        f.write(cfm_str)

                if best_valid_loss <= valid_loss_per_epoch:
                    earlystop_patience += 1
                    decay_patience += 1
                else:
                    best_valid_acc = valid_acc_per_epoch
                    best_valid_loss = valid_loss_per_epoch
                    earlystop_patience = 0
                    decay_patience = 0

                #todo Decay learning rate
                # if decay_patience >= flags.FLAGS.decay_patience:
                #     if train_model.lr > 0.00001:
                #         train_model.lr = max(train_model.lr / 2, 0.00001)

                #todo Early stopping
                if earlystop_patience >= flags.FLAGS.earlystop_patience:
                    print('>> Early stopping occurred in ' + str(epoch) + ' epoch.')
                    break

            tot_train_et = time.time()
            print('>> Learning is complete. The total learning time is %d second.' % (tot_train_et - tot_train_st))

    def roc_curve(self, labels, probs, epoch):
        roc_per_class = defaultdict(list)

        for cls in range(flags.FLAGS.image_class):
            roc_per_class[cls].append([])
            roc_per_class[cls].append([])

            for label, prob in zip(labels, probs):
                if cls == label:
                    roc_per_class[cls][0].append(1)
                    roc_per_class[cls][1].append(prob[label])
                else:
                    roc_per_class[cls][0].append(0)
                    roc_per_class[cls][1].append(1. - prob[label])

        for key in roc_per_class.keys():
            os.makedirs(os.path.join(flags.FLAGS.roc_curve_log_path, str(self.seq), str(epoch)), exist_ok=True)

            labels_per_cls, probs_per_cls = roc_per_class[key][0], roc_per_class[key][1]

            fpr_train, tpr_train, thresholds_train = roc_curve(y_true=labels_per_cls, y_score=probs_per_cls, pos_label=True)
            sum_sensitivity_specificity_train = tpr_train + (1 - fpr_train)
            best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
            best_threshold = thresholds_train[best_threshold_id_train]
            best_fpr_train = fpr_train[best_threshold_id_train]
            best_tpr_train = tpr_train[best_threshold_id_train]
            y_train = probs_per_cls > best_threshold

            cm_train = confusion_matrix(y_true=labels_per_cls, y_pred=y_train)
            acc_train = accuracy_score(y_true=labels_per_cls, y_pred=y_train)
            auc_train = roc_auc_score(y_true=labels_per_cls, y_score=y_train)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            curve1 = ax.plot(fpr_train, tpr_train, color='orange', linewidth=5)
            curve2 = ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            dot = ax.plot(best_fpr_train, best_tpr_train, marker='o', color='black')
            ax.text(best_fpr_train, best_tpr_train, s='(%.3f,%.3f)' % (best_fpr_train, best_tpr_train))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.suptitle('ROC curve, AUC = %.4f' % auc_train, fontsize=20)

            fig.savefig(os.path.join(flags.FLAGS.roc_curve_log_path, str(self.seq), str(epoch), str(key) + '.png'))

    def test(self):
        tf.reset_default_graph()

        self._loader.init_test()

        test_step_num = self._loader.valid_len // flags.FLAGS.batch_size

        test_normal_batch = self._loader.valid_loader()

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True,
                                      per_process_gpu_memory_fraction=0.5)
        )

        with tf.Session(config=config) as sess:
            test_model = Model(sess=sess, is_training=False, is_tb_logging=False, name=self.name)

            #todo Saver 생성
            self._test_saver = tf.train.Saver(var_list=tf.global_variables(self.name))

            #todo 학습된 체크포인트 파일 로드
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.train_log_path, str(self.seq)))

            if ckpt_st is not None:
                self._test_saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored from', ckpt_st.model_checkpoint_path)

            print('>> Tensorflow session built. Variables initialized.')

            # todo 텐서플로우 그래프 저장
            tf.train.write_graph(sess.graph_def, os.path.join(flags.FLAGS.deploy_log_path, str(self.seq)), 'test_graph.pbtxt')

            print('>> Tensorflow graph saved.')

            #todo Total Testing
            print('>> Start testing.')

            tot_test_filename = []
            tot_test_prob = []
            test_cls = {}

            for (path, dirs, files) in os.walk(os.path.join(flags.FLAGS.data_path, 'test')):
                for file in files:
                    tot_test_filename.append(file)

            for key in self._loader.cls.keys():
                test_cls[self._loader.cls[key]] = key

            for step in range(1, test_step_num + 1):
                test_data = sess.run(test_normal_batch)
                test_batch_x = test_data[0]

                test_prob = test_model.test(test_batch_x)

                tot_test_prob.append(test_prob)

            with open(os.path.join(flags.FLAGS.deploy_log_path, str(self.seq), 'mobilenetv2.txt'), mode='w') as f:
                for fname, prob in zip(np.asarray(tot_test_filename), np.argmax(np.asarray(tot_test_prob), axis=-1).flatten()):
                    f.write(str(fname) + ' ' + str(prob) + '\n')

            #todo Checkpoint file logging
            if self.is_ckpt_logging:
                self._test_saver.save(sess, os.path.join(flags.FLAGS.deploy_log_path, str(self.seq), 'test_tiny_imagenet'))

    def visualization(self, sample_per_class):
        def save_cam_img(cam_outputs, cam_y, cam_prob):
            f = plt.figure(figsize=(10, 8))
            plt.suptitle('Grad CAM (Gradient-weighted Class Activation Mapping)', fontsize=20)
            outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)

            inner = gridspec.GridSpecFromSubplotSpec(10, sample_per_class, subplot_spec=outer[0], wspace=0.1, hspace=0.8)

            for cls in range(10):
                for sample in range(sample_per_class):
                    subplot = plt.Subplot(f, inner[sample + cls * sample_per_class])
                    subplot.axis('off')
                    subplot.imshow(cam_outputs[sample + cls * sample_per_class])
                    subplot.set_title(str(cam_y[sample + cls * sample_per_class]) + ' / ' + str(cam_prob[sample + cls * sample_per_class]))
                    f.add_subplot(subplot)

            f.savefig(os.path.join(flags.FLAGS.cam_log_path, str(self.seq), 'cam_test.png'))
            print('>> Grad CAM Complete')

        def normal_data(filename, y):
            with tf.variable_scope(name_or_scope='cam_normal'):
                x = tf.read_file(filename=filename)
                x = tf.image.decode_png(contents=x, channels=3, name='decode_png')
                x = tf.divide(tf.cast(x, tf.float32), 255.)
            return x, y, filename

        def data_loader():
            with tf.variable_scope('cam_loader'):
                dataset = tf.data.Dataset.from_tensor_slices((self._loader.valid_x, self._loader.valid_y)).repeat()

                dataset_map = dataset.map(normal_data).batch(flags.FLAGS.image_class * sample_per_class)
                iterator = dataset_map.make_one_shot_iterator()
                batch_input = iterator.get_next()

            return batch_input

        tf.reset_default_graph()

        self._loader.init_validation()

        cam_step_num = self._loader.valid_len // flags.FLAGS.batch_size
        cam_normal_batch = data_loader()
        cam_sample_data = dict()

        print('>> Start Grad-CAM ')

        with tf.Session() as sess:
            for step in range(1, cam_step_num + 1):
                batch_x, batch_y, batch_fname = sess.run(cam_normal_batch)
                cnt = 0

                for x, y, fname in zip(batch_x, batch_y, batch_fname):
                    if cam_sample_data.get(y) is None:
                        cam_sample_data[y] = [[], []]

                    if len(cam_sample_data[y][0]) != sample_per_class:
                        cam_sample_data[y][0].append(x)
                        cam_sample_data[y][1].append(fname)

                    for key in cam_sample_data.keys():
                        if len(cam_sample_data[key][0]) == sample_per_class:
                            cnt += 1

                if cnt == flags.FLAGS.image_class:
                    break

            model = Model(sess=sess, is_training=False, is_tb_logging=False, name=self.name)

            cam = GradCAM(instance=model, sample_size=10 * sample_per_class, name='grad_cam')
            cam.build()

            # todo Saver 생성
            self._cam_saver = tf.train.Saver(var_list=tf.global_variables())

            # todo 학습된 체크포인트 파일 로드
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.train_log_path, str(self.seq)))

            if ckpt_st is not None:
                self._cam_saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored from', ckpt_st.model_checkpoint_path)

            print('>> Tensorflow session built. Variables initialized.')

            # todo 텐서플로우 그래프 저장
            tf.train.write_graph(sess.graph_def, os.path.join(flags.FLAGS.cam_log_path, str(self.seq)), 'cam_graph.pbtxt')

            print('>> Tensorflow graph saved.')

            cam_batch_x, cam_batch_y, cam_batch_fname = [], [], []

            sample_cls = np.random.permutation(flags.FLAGS.image_class)[:10]

            for key in sample_cls:
                temp = cam_sample_data.get(key)
                for x, fname in zip(temp[0], temp[1]):
                    cam_batch_x.append(x)
                    cam_batch_y.append(key)
                    cam_batch_fname.append(fname)

            cam_batch_x = np.asarray(cam_batch_x)
            cam_batch_y = np.asarray(cam_batch_y)
            cam_batch_fname = np.asarray(cam_batch_fname)

            cam_outputs, cam_prob = cam.visualize(x=cam_batch_x, file_names=cam_batch_fname)

            save_cam_img(cam_outputs, cam_batch_y, cam_prob)

            self._cam_saver.save(sess, os.path.join(flags.FLAGS.cam_log_path, str(self.seq), 'cam_tiny_imagenet'))

trainer = Trainer(seq=9,
                  is_db_logging=True,
                  is_ckpt_logging=True,
                  is_cfm_logging=True,
                  is_roc_logging=True,
                  is_tb_logging=True,
                  dataset='tiny_imagenet',
                  name='inception_resnet_v2')
trainer.train()
trainer.test()
trainer.visualization(sample_per_class=5)