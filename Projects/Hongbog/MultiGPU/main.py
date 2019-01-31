import os
import time

from Projects.Hongbog.MultiGPU.constants import *
from Projects.Hongbog.MultiGPU.data_loader import DataLoader
from Projects.Hongbog.MultiGPU.multi_gpu import MultiGPU

"""
    ▣ ShakeNet 신경망
"""
class Trainer:

    def __init__(self, seq, is_db_logging, is_ckpt_logging, is_cfm_logging, is_roc_logging, is_tb_logging, dataset, name):
        self.seq = seq
        self.best_epoch = -1
        self.is_db_logging = is_db_logging
        self.is_ckpt_logging = is_ckpt_logging
        self.is_cfm_logging = is_cfm_logging
        self.is_roc_logging = is_roc_logging
        self.is_tb_logging = is_tb_logging
        self.dataset = dataset
        self.name = name

        self._loader = DataLoader()
        print('>> The data loader has been initialized.')

        self._utils = MultiGPU()
        print('>> MultiGPU class has been initialized.')

    def train(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

            #todo Data Loader Initialization
            self._loader.init_train()
            self._loader.init_test()

            train_step_num = int(self._loader.train_len / flags.FLAGS.batch_size / flags.FLAGS.num_gpus)
            # test_step_num = int(self._loader.test_len / flags.FLAGS.batch_size / flags.FLAGS.num_gpus)

            train_x, train_y = self._loader.train_loader()
            # test_x, test_y = self._loader.test_loader()

            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0), trainable=False)

            decay_lr = tf.train.cosine_decay(flags.FLAGS.lr, global_step, train_step_num)

            opt = tf.train.RMSPropOptimizer(learning_rate=decay_lr, momentum=0.9, epsilon=1.0)

            grads, loss, summaries = self._utils.init_tower(num_gpus=flags.FLAGS.num_gpus,
                                                            train_x=train_x,
                                                            train_y=train_y,
                                                            opt=opt)

            summaries.append(tf.summary.scalar('lr', decay_lr))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            update_ops.append(opt.apply_gradients(grads, global_step=global_step))

            update_ops = tf.group(*update_ops)

            with tf.control_dependencies([update_ops]):
                train_tensor = tf.identity(loss)

            saver = tf.train.Saver(tf.global_variables())

            summary_op = tf.summary.merge(summaries)

            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True,  # GPU 메모리 증분 할당
                                          per_process_gpu_memory_fraction=1.0),  # GPU 당 할당할 메모리 양
                allow_soft_placement=True,
                log_device_placement=False
            )

            tot_start_time = time.time()
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                tf.train.start_queue_runners(sess=sess)

                summary_writer = tf.summary.FileWriter(flags.FLAGS.tensorboard_log_path, sess.graph)

                for step in range(flags.FLAGS.epochs * train_step_num):
                    step_start_time = time.time()
                    _, step_loss = sess.run([train_tensor, loss])
                    step_end_time = time.time()
                    print('[Step-%d], loss: %.4f, time: %.3f' % (step, step_loss, (step_end_time-step_start_time)/flags.FLAGS.num_gpus))

                    if step % 100 == 0:
                        summary_writer.add_summary(sess.run(summary_op), step)

                    if step % 1000 == 0 or (step + 1) == flags.FLAGS.epochs * train_step_num:
                        saver.save(sess, os.path.join(flags.FLAGS.train_log_path, 'model.ckpt'), global_step=step)
            tot_end_time = time.time()

            print('>>> Total Train Time: %.3f' % (tot_end_time - tot_start_time))













    #         #todo Model 객체 생성
    #         train_model = Model(sess=sess, is_tb_logging=self.is_tb_logging, name=self.name)
    #
    #         sess.run(tf.global_variables_initializer())
    #
    #         print('>> Tensorflow session built. Variables initialized.')
    #
    #         #todo 훈련 시 필요한 로그 디렉토리 생성
    #         os.makedirs(os.path.join(flags.FLAGS.train_log_path, str(self.seq)), exist_ok=True)
    #         os.makedirs(os.path.join(flags.FLAGS.deploy_log_path, str(self.seq)), exist_ok=True)
    #         os.makedirs(os.path.join(flags.FLAGS.cam_log_path, str(self.seq)), exist_ok=True)
    #         os.makedirs(os.path.join(flags.FLAGS.tensorboard_log_path, str(self.seq)), exist_ok=True)
    #         os.makedirs(os.path.join(flags.FLAGS.roc_curve_log_path, str(self.seq)), exist_ok=True)
    #         os.makedirs(os.path.join(flags.FLAGS.cfm_log_path, str(self.seq)), exist_ok=True)
    #
    #         #todo 텐서플로우 그래프 저장
    #         tf.train.write_graph(sess.graph_def, os.path.join(flags.FLAGS.train_log_path, str(self.seq)), 'train_graph.pbtxt')
    #
    #         print('>> Tensorflow graph saved.')
    #
    #         #todo Saver 생성
    #         self._train_saver = tf.train.Saver(var_list=tf.global_variables(self.name), max_to_keep=flags.FLAGS.epoch)
    #
    #         #todo 텐서보드 로깅을 위한 FileWriter 생성
    #         if self.is_tb_logging:
    #             tensorboard_writer = tf.summary.FileWriter(os.path.join(flags.FLAGS.tensorboard_log_path, str(self.seq)), graph=tf.get_default_graph())
    #
    #         best_test_acc, best_test_loss = 0., math.inf
    #         earlystop_patience, decay_patience = 0, 0
    #
    #         #todo Total Train/testation
    #         print('>> Start learning.')
    #         tot_train_st = time.time()
    #         for epoch in range(1, flags.FLAGS.epoch + 1):
    #             train_acc_per_epoch, train_loss_per_epoch = [], []
    #             test_acc_per_epoch, test_loss_per_epoch = [], []
    #             test_prob_per_epoch, test_label_per_epoch = [], []
    #
    #             #todo Train per epoch
    #             print('>> Epoch [%d]' % (epoch))
    #             epoch_train_st = time.time()
    #             for step in range(1, train_step_num + 1):
    #                 train_data = sess.run([train_normal_batch, train_random_crop_batch])
    #                 train_batch_x, train_batch_y = np.concatenate([data[0] for data in train_data]), np.concatenate([data[1] for data in train_data])
    #
    #                 train_acc_per_step, train_loss_per_step = [], []
    #                 step_train_st = time.time()
    #                 for idx in range(0, flags.FLAGS.batch_size * 2, flags.FLAGS.batch_size):  # 2: augmentation 개수
    #                     if self.is_tb_logging:
    #                         train_acc, train_loss, train_summary, _ = \
    #                             train_model.train(x=train_batch_x[idx: idx + flags.FLAGS.batch_size],
    #                                               y=train_batch_y[idx: idx + flags.FLAGS.batch_size])
    #                     else:
    #                         train_acc, train_loss, _ = \
    #                             train_model.train(x=train_batch_x[idx: idx + flags.FLAGS.batch_size],
    #                                               y=train_batch_y[idx: idx + flags.FLAGS.batch_size])
    #
    #                     train_acc_per_step.append(train_acc)
    #                     train_loss_per_step.append(train_loss)
    #                     train_acc_per_epoch.append(train_acc)
    #                     train_loss_per_epoch.append(train_loss)
    #
    #
    #                 step_train_et = time.time()
    #
    #                 train_acc_per_step = np.sum(train_acc_per_step) / len(train_acc_per_step)
    #                 train_loss_per_step = np.sum(train_loss_per_step) / len(train_loss_per_step)
    #
    #                 print('>> [Step-Train] epoch/step [%d/%d], acc: %.6f, loss: %.6f, time: %.2f'
    #                       % (epoch, step, train_acc_per_step, train_loss_per_step, (step_train_et - step_train_st)))
    #             epoch_train_et = time.time()
    #
    #             # todo Tensorboard logging
    #             if self.is_tb_logging:
    #                 tensorboard_writer.add_summary(summary=train_summary, global_step=epoch)
    #
    #
    #             # todo Checkpoint file logging
    #             if self.is_ckpt_logging:
    #                 self._train_saver.save(sess, os.path.join(flags.FLAGS.train_log_path, str(self.seq), 'train_eyeoclock'), global_step=epoch)
    #
    #             # todo Test per epoch
    #             epoch_test_st = time.time()
    #             for step in range(1, test_step_num + 1):
    #                 test_data = sess.run(test_normal_batch)
    #                 test_batch_x, test_batch_y = test_data[0], test_data[1]
    #
    #                 test_acc, test_loss, test_prob = train_model.test(test_batch_x, test_batch_y)
    #
    #                 test_acc_per_epoch.append(test_acc)
    #                 test_loss_per_epoch.append(test_loss)
    #                 test_prob_per_epoch.append(test_prob)
    #                 test_label_per_epoch.append(test_batch_y)
    #             epoch_test_et = time.time()
    #
    #             train_acc_per_epoch = np.sum(train_acc_per_epoch) / len(train_acc_per_epoch)
    #             train_loss_per_epoch = np.sum(train_loss_per_epoch) / len(train_loss_per_epoch)
    #             test_acc_per_epoch = np.sum(test_acc_per_epoch) / len(test_acc_per_epoch)
    #             test_loss_per_epoch = np.sum(test_loss_per_epoch) / len(test_loss_per_epoch)
    #
    #             print('>> [Epoch-Train] epoch: [%d], acc: %.6f, loss: %.6f, time: %.2f'
    #                   % (epoch, train_acc_per_epoch, train_loss_per_epoch, (epoch_train_et - epoch_train_st)))
    #             print('>> [Epoch-Test] epoch: [%d], acc: %.6f, loss: %.6f, time: %.2f'
    #                   % (epoch, test_acc_per_epoch, test_loss_per_epoch, (epoch_test_et - epoch_test_st)))
    #
    #             #todo Database logging
    #             if self.is_db_logging:
    #                 self.db.mon_data_to_db(self.dataset, self.name, 'train', self.seq, epoch, train_acc_per_epoch, train_loss_per_epoch, (epoch_train_et - epoch_train_st))
    #                 self.db.mon_data_to_db(self.dataset, self.name, 'test', self.seq, epoch, test_acc_per_epoch, test_loss_per_epoch, (epoch_test_et - epoch_test_st))
    #
    #             #todo Confusion matrix
    #             print('>> [Confusion-Matrix] Epoch %d -' % (epoch))
    #             print(sess.run(tf.confusion_matrix(labels=np.asarray(test_label_per_epoch).flatten(),
    #                                                predictions=np.argmax(np.asarray(test_prob_per_epoch), axis=-1).flatten(),
    #                                                num_classes=flags.FLAGS.image_class)))
    #             if self.is_cfm_logging:
    #                 cfm_str = ''
    #                 for line in sess.run(tf.confusion_matrix(labels=np.asarray(test_label_per_epoch).flatten(),
    #                                                          predictions=np.argmax(np.asarray(test_prob_per_epoch), axis=-1).flatten(),
    #                                                          num_classes=flags.FLAGS.image_class)):
    #                     temp = ''
    #                     for element in line:
    #                         fill_str = str(element).replace(' ', '').zfill(4)
    #                         for idx, ch in enumerate(fill_str):
    #                             if ch == '0' and idx < 3:
    #                                 temp += ' '
    #                             else:
    #                                 temp += fill_str[idx:]
    #                                 break
    #                         temp += ' '
    #                     cfm_str = cfm_str + temp + '\n'
    #
    #                 with open(os.path.join(flags.FLAGS.cfm_log_path, str(self.seq), str(epoch) + '.txt'), mode='w') as f:
    #                     f.write(cfm_str)
    #
    #             #todo ROC Curve
    #             if self.is_roc_logging and (epoch % 10 == 0):
    #                 self.roc_curve(labels=np.asarray(test_label_per_epoch).flatten(),
    #                                probs=np.asarray(test_prob_per_epoch).reshape((-1, flags.FLAGS.image_class)),
    #                                epoch=epoch)
    #
    #             if best_test_loss <= test_loss_per_epoch:
    #                 earlystop_patience += 1
    #                 decay_patience += 1
    #             else:
    #                 best_test_acc = test_acc_per_epoch
    #                 best_test_loss = test_loss_per_epoch
    #                 earlystop_patience = 0
    #                 decay_patience = 0
    #                 self.best_epoch = epoch
    #
    #             #todo Decay learning rate
    #             # if decay_patience >= flags.FLAGS.decay_patience:
    #             #     if train_model.lr > 0.00001:
    #             #         train_model.lr = max(train_model.lr / 2, 0.00001)
    #
    #             #todo Early stopping
    #             # if earlystop_patience >= flags.FLAGS.earlystop_patience:
    #             #     print('>> Early stopping occurred in ' + str(epoch) + ' epoch.')
    #             #     self.roc_curve(labels=np.asarray(test_label_per_epoch).flatten(),
    #             #                    probs=np.asarray(test_prob_per_epoch).reshape((-1, flags.FLAGS.image_class)),
    #             #                    epoch=epoch)
    #             #     break
    #
    #         tot_train_et = time.time()
    #         print('>> Learning is complete. The total learning time is %d second.' % (tot_train_et - tot_train_st))
    #
    # def roc_curve(self, labels, probs, epoch):
    #     roc_per_class = defaultdict(list)
    #
    #     for cls in range(flags.FLAGS.image_class):
    #         roc_per_class[cls].append([])
    #         roc_per_class[cls].append([])
    #
    #         for label, prob in zip(labels, probs):
    #             if cls == label:
    #                 roc_per_class[cls][0].append(1)
    #                 roc_per_class[cls][1].append(prob[label])
    #             else:
    #                 roc_per_class[cls][0].append(0)
    #                 roc_per_class[cls][1].append(1. - prob[label])
    #
    #     for key in roc_per_class.keys():
    #         os.makedirs(os.path.join(flags.FLAGS.roc_curve_log_path, str(self.seq), str(epoch)), exist_ok=True)
    #
    #         labels_per_cls, probs_per_cls = roc_per_class[key][0], roc_per_class[key][1]
    #
    #         fpr_train, tpr_train, thresholds_train = roc_curve(y_true=labels_per_cls, y_score=probs_per_cls, pos_label=True)
    #         sum_sensitivity_specificity_train = tpr_train + (1 - fpr_train)
    #         best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
    #         best_threshold = thresholds_train[best_threshold_id_train]
    #         best_fpr_train = fpr_train[best_threshold_id_train]
    #         best_tpr_train = tpr_train[best_threshold_id_train]
    #         y_train = probs_per_cls > best_threshold
    #
    #         cm_train = confusion_matrix(y_true=labels_per_cls, y_pred=y_train)
    #         acc_train = accuracy_score(y_true=labels_per_cls, y_pred=y_train)
    #         auc_train = roc_auc_score(y_true=labels_per_cls, y_score=y_train)
    #
    #         fig = plt.figure(figsize=(10, 8))
    #         ax = fig.add_subplot(111)
    #         curve1 = ax.plot(fpr_train, tpr_train, color='orange', linewidth=5)
    #         curve2 = ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    #         dot = ax.plot(best_fpr_train, best_tpr_train, marker='o', color='black')
    #         ax.text(best_fpr_train, best_tpr_train, s='(%.3f,%.3f)' % (best_fpr_train, best_tpr_train))
    #         plt.xlim([0.0, 1.0])
    #         plt.ylim([0.0, 1.0])
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.suptitle('ROC curve, AUC = %.4f' % auc_train, fontsize=20)
    #
    #         fig.savefig(os.path.join(flags.FLAGS.roc_curve_log_path, str(self.seq), str(epoch), str(key) + '.png'))
    #
    # def test(self):
    #     tf.reset_default_graph()
    #
    #     self._loader.init_test()
    #
    #     test_step_num = self._loader.test_len // flags.FLAGS.batch_size
    #
    #     test_normal_batch = self._loader.test_loader()
    #
    #     config = tf.ConfigProto(
    #         gpu_options=tf.GPUOptions(allow_growth=True,
    #                                   per_process_gpu_memory_fraction=0.4)
    #     )
    #
    #     with tf.Session(config=config) as sess:
    #         test_model = Model(sess=sess, is_tb_logging=False, name=self.name)
    #
    #         #todo Saver 생성
    #         self._test_saver = tf.train.Saver(var_list=tf.global_variables(self.name))
    #
    #         #todo 학습된 체크포인트 파일 로드
    #         # if self.best_epoch == -1:
    #         #     ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.train_log_path, str(self.seq)))
    #         #
    #         #     if ckpt_st is not None:
    #         #         self._test_saver.restore(sess, ckpt_st.model_checkpoint_path)
    #         #         print('>> Model Restored from', ckpt_st.model_checkpoint_path)
    #         # else:
    #         #     checkpoint_file = os.path.join(flags.FLAGS.train_log_path, str(self.seq), 'train_eyeoclock-' + str(self.best_epoch))
    #         #     self._test_saver.restore(sess, checkpoint_file)
    #         #     print('>> Model Restored from', checkpoint_file)
    #
    #         checkpoint_file = os.path.join(flags.FLAGS.train_log_path, str(self.seq), 'train_eyeoclock-258')
    #         self._test_saver.restore(sess, checkpoint_file)
    #         print('>> Model Restored from', checkpoint_file)
    #
    #         print('>> Tensorflow session built. Variables initialized.')
    #
    #         # todo 텐서플로우 그래프 저장
    #         tf.train.write_graph(sess.graph_def, os.path.join(flags.FLAGS.deploy_log_path, str(self.seq)), 'test_graph.pbtxt')
    #
    #         print('>> Tensorflow graph saved.')
    #
    #         #todo Total Testing
    #         print('>> Start testing.')
    #
    #         tot_test_acc, tot_test_loss = [], []
    #         tot_test_prob, tot_test_label = [], []
    #         tot_test_st = time.time()
    #
    #         for step in range(1, test_step_num + 1):
    #             test_data = sess.run(test_normal_batch)
    #             test_batch_x, test_batch_y = test_data[0], test_data[1]
    #
    #             test_acc, test_loss, test_prob = test_model.test(test_batch_x, test_batch_y)
    #
    #             tot_test_acc.append(test_acc)
    #             tot_test_loss.append(test_loss)
    #             tot_test_prob.append(test_prob)
    #             tot_test_label.append(test_batch_y)
    #
    #         tot_test_et = time.time()
    #
    #         tot_test_acc = np.sum(tot_test_acc) / len(tot_test_acc)
    #         tot_test_loss = np.sum(tot_test_loss) / len(tot_test_loss)
    #
    #         print('>> [Test] acc: %.6f, loss: %.6f, time: %.2f'
    #               % (tot_test_acc, tot_test_loss, (tot_test_et - tot_test_st)))
    #
    #         #todo Database logging
    #         if self.is_db_logging:
    #             self.db.mon_data_to_db(self.dataset, self.name, 'tot_test', self.seq, 1, tot_test_acc, tot_test_loss, (tot_test_et - tot_test_st))
    #
    #         #todo Checkpoint file logging
    #         if self.is_ckpt_logging:
    #             self._test_saver.save(sess, os.path.join(flags.FLAGS.deploy_log_path, str(self.seq), 'test_eyeoclock'))
    #
    #         #todo Confusion matrix
    #         print('>> [Confusion-Matrix] test -')
    #         print(sess.run(tf.confusion_matrix(labels=np.asarray(tot_test_label).flatten(),
    #                                            predictions=np.argmax(np.asarray(tot_test_prob), axis=-1).flatten(),
    #                                            num_classes=flags.FLAGS.image_class)))
    #         if self.is_cfm_logging:
    #             cfm_str = ''
    #             for line in sess.run(tf.confusion_matrix(labels=np.asarray(tot_test_label).flatten(),
    #                                                      predictions=np.argmax(np.asarray(tot_test_prob), axis=-1).flatten(),
    #                                                      num_classes=flags.FLAGS.image_class)):
    #                 temp = ''
    #                 for element in line:
    #                     fill_str = str(element).replace(' ', '').zfill(4)
    #                     for idx, ch in enumerate(fill_str):
    #                         if ch == '0' and idx < 3:
    #                             temp += ' '
    #                         else:
    #                             temp += fill_str[idx:]
    #                             break
    #                     temp += ' '
    #                 cfm_str = cfm_str + temp + '\n'
    #
    #             with open(os.path.join(flags.FLAGS.cfm_log_path, str(self.seq), 'test.txt'), mode='w') as f:
    #                 f.write(cfm_str)
    #
    #         #todo ROC Curve
    #         if self.is_roc_logging:
    #             self.roc_curve(labels=np.asarray(tot_test_label).flatten(),
    #                            probs=np.asarray(tot_test_prob).reshape((-1, flags.FLAGS.image_class)),
    #                            epoch='test')
    #
    # def visualization(self, sample_per_class):
    #     def save_cam_img(cam_outputs, cam_y, cam_prob):
    #         f = plt.figure(figsize=(10, 8))
    #         plt.suptitle('Grad CAM (Gradient-weighted Class Activation Mapping)', fontsize=20)
    #         outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
    #
    #         inner = gridspec.GridSpecFromSubplotSpec(flags.FLAGS.image_class, sample_per_class, subplot_spec=outer[0], wspace=0.1, hspace=0.8)
    #
    #         for cls in range(flags.FLAGS.image_class):
    #             for sample in range(sample_per_class):
    #                 subplot = plt.Subplot(f, inner[sample + cls * sample_per_class])
    #                 subplot.axis('off')
    #                 subplot.imshow(cam_outputs[sample + cls * sample_per_class])
    #                 subplot.set_title(str(cam_y[sample + cls * sample_per_class]) + ' / ' + str(cam_prob[sample + cls * sample_per_class]))
    #                 f.add_subplot(subplot)
    #
    #         f.savefig(os.path.join(flags.FLAGS.cam_log_path, str(self.seq), 'cam_test.png'))
    #         print('>> Grad CAM Complete')
    #
    #     def normal_data(filename, y):
    #         with tf.variable_scope(name_or_scope='cam_normal'):
    #             x = tf.read_file(filename=filename)
    #             x = tf.image.decode_png(contents=x, channels=3, name='decode_png')
    #             x = tf.divide(tf.cast(x, tf.float32), 255.)
    #         return x, y, filename
    #
    #     def data_loader():
    #         with tf.variable_scope('cam_loader'):
    #             dataset = tf.data.Dataset.from_tensor_slices((self._loader.test_x, self._loader.test_y)).repeat()
    #
    #             dataset_map = dataset.map(normal_data).batch(flags.FLAGS.image_class * sample_per_class)
    #             iterator = dataset_map.make_one_shot_iterator()
    #             batch_input = iterator.get_next()
    #
    #         return batch_input
    #
    #     tf.reset_default_graph()
    #
    #     self._loader.init_test()
    #
    #     cam_step_num = (self._loader.test_len // flags.FLAGS.batch_size) * flags.FLAGS.image_class
    #     cam_normal_batch = data_loader()
    #     cam_sample_data = dict()
    #
    #     print('>> Start Grad-CAM ')
    #
    #     with tf.Session() as sess:
    #         for step in range(1, cam_step_num + 1):
    #             batch_x, batch_y, batch_fname = sess.run(cam_normal_batch)
    #             cnt = 0
    #
    #             for x, y, fname in zip(batch_x, batch_y, batch_fname):
    #                 if cam_sample_data.get(y) is None:
    #                     cam_sample_data[y] = [[], []]
    #
    #                 if len(cam_sample_data[y][0]) != sample_per_class:
    #                     cam_sample_data[y][0].append(x)
    #                     cam_sample_data[y][1].append(fname)
    #
    #                 for key in cam_sample_data.keys():
    #                     if len(cam_sample_data[key][0]) == sample_per_class:
    #                         cnt += 1
    #
    #             if cnt == flags.FLAGS.image_class:
    #                 break
    #
    #         model = Model(sess=sess, is_tb_logging=False, name=self.name)
    #
    #         cam = GradCAM(instance=model, sample_size=flags.FLAGS.image_class * sample_per_class, name='grad_cam')
    #         cam.build()
    #
    #         # todo Saver 생성
    #         self._cam_saver = tf.train.Saver(var_list=tf.global_variables())
    #
    #         # todo 학습된 체크포인트 파일 로드
    #         ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.deploy_log_path, str(self.seq)))
    #
    #         if ckpt_st is not None:
    #             self._cam_saver.restore(sess, ckpt_st.model_checkpoint_path)
    #             print('>> Model Restored from', ckpt_st.model_checkpoint_path)
    #
    #         print('>> Tensorflow session built. Variables initialized.')
    #
    #         # todo 텐서플로우 그래프 저장
    #         tf.train.write_graph(sess.graph_def, os.path.join(flags.FLAGS.cam_log_path, str(self.seq)), 'cam_graph.pbtxt')
    #
    #         print('>> Tensorflow graph saved.')
    #
    #         cam_batch_x, cam_batch_y, cam_batch_fname = [], [], []
    #
    #         for key in cam_sample_data.keys():
    #             temp = cam_sample_data.get(key)
    #             for x, fname in zip(temp[0], temp[1]):
    #                 cam_batch_x.append(x)
    #                 cam_batch_y.append(key)
    #                 cam_batch_fname.append(fname)
    #
    #         cam_batch_x = np.asarray(cam_batch_x)
    #         cam_batch_y = np.asarray(cam_batch_y)
    #         cam_batch_fname = np.asarray(cam_batch_fname)
    #
    #         cam_outputs, cam_prob = cam.visualize(x=cam_batch_x, file_names=cam_batch_fname)
    #
    #         save_cam_img(cam_outputs, cam_batch_y, cam_prob)
    #
    #         self._cam_saver.save(sess, os.path.join(flags.FLAGS.cam_log_path, str(self.seq), 'cam_eyeoclock'))

trainer = Trainer(seq=4,
                  is_db_logging=True,
                  is_ckpt_logging=True,
                  is_cfm_logging=False,
                  is_roc_logging=False,
                  is_tb_logging=True,
                  dataset='cifar10',
                  name='nasnet-a')
trainer.train()
# trainer.test()
# trainer.visualization(sample_per_class=5)