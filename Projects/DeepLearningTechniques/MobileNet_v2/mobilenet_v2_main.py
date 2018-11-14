from Projects.DeepLearningTechniques.MobileNet_v2.constants import *
from Projects.DeepLearningTechniques.MobileNet_v2.dataloader import DataLoader
from Projects.DeepLearningTechniques.MobileNet_v2.mobilenet_v2_model import *
from Projects.DeepLearningTechniques.MobileNet_v2.cam import GradCAM, GuidedGradCAM

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Neuralnet:
    def __init__(self, is_training, save_type=None):
        self.is_training = is_training
        self.save_type = save_type

    def train(self):
        loader = DataLoader(batch_size=flags.FLAGS.batch_size, data_path=flags.FLAGS.data_path)
        print('>> DataLoader created')

        train_num = loader.train_len // flags.FLAGS.batch_size
        # test_num = loader.test_len // flags.FLAGS.batch_size
        train_batch1, train_batch2, train_batch3, train_batch4, train_batch5, train_batch6 = loader.train_loader()
        # test_batch = loader.test_loader()

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1)
        )

        with tf.Session(config=config) as sess:
            model = Model(sess=sess, is_training=self.is_training, name='model')
            model.build()

            print('>> Tensorflow session built. Variables initialized.')
            sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()

            ckpt_st = tf.train.get_checkpoint_state(flags.FLAGS.trained_param_path)

            if ckpt_st is not None:
                # self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            '''텐서플로우 그래프 저장'''
            tf.train.write_graph(sess.graph_def, flags.FLAGS.trained_param_path, 'graph.pbtxt')
            print('>> Graph saved')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('>> Running started.')

            for epoch in range(1, flags.FLAGS.epochs+1):
                tot_train_acc, tot_train_loss = [], []
                tot_test_acc, tot_test_loss = [],[]

                '''Model Train'''
                train_st = time.time()
                for step in range(1, train_num+1):
                    train_data_1, train_data_2, train_data_3, train_data_4, train_data_5, train_data_6 = \
                        sess.run([train_batch1, train_batch2, train_batch3, train_batch4, train_batch5, train_batch6])
                    train_batch_x = np.concatenate([train_data_1[0], train_data_2[0], train_data_3[0], train_data_4[0], train_data_5[0], train_data_6[0]])
                    train_batch_y = np.concatenate([train_data_1[1], train_data_2[1], train_data_3[1], train_data_4[1], train_data_5[1], train_data_6[1]])
                    # train_batch_x = np.concatenate([train_data_1[0], train_data_3[0], train_data_4[0], train_data_5[0], train_data_6[0]])
                    # train_batch_y = np.concatenate([train_data_1[1], train_data_3[1], train_data_4[1], train_data_5[1], train_data_6[1]])

                    st = time.time()
                    step_train_acc, step_train_loss = [], []

                    for idx in range(0, 60, flags.FLAGS.batch_size):
                        train_acc, train_loss, _ = model.train(x=train_batch_x[idx:idx+flags.FLAGS.batch_size],
                                                               y=train_batch_y[idx:idx+flags.FLAGS.batch_size])
                        step_train_acc.append(train_acc)
                        step_train_loss.append(train_loss)
                        tot_train_acc.append(train_acc)
                        tot_train_loss.append(train_loss)

                    et = time.time()

                    step_train_acc = float(np.mean(np.array(step_train_acc)))
                    step_train_loss = float(np.mean(np.array(step_train_loss)))
                    print(">> [Step-Train] epoch/step: [%d/%d], Accuracy: %.6f, Loss: %.6f, step_time: %.2f"
                          % (epoch, step, step_train_acc, step_train_loss, et - st))
                train_et = time.time()
                tot_train_time = train_et - train_st

                '''Model Test'''
                # for step in range(1, test_num + 1):
                #     test_data = sess.run(test_batch)
                #     test_batch_x, test_batch_y = test_data
                #
                #     test_acc, test_loss = model.validate(x=test_batch_x, y=test_batch_y)
                #
                #     tot_test_acc.append(test_acc)
                #     tot_test_loss.append(test_loss)

                tot_train_acc = float(np.mean(np.array(tot_train_acc)))
                tot_train_loss = float(np.mean(np.array(tot_train_loss)))

                # tot_test_acc = float(np.mean(np.array(tot_test_acc)))
                # tot_test_loss = float(np.mean(np.array(tot_test_loss)))

                print('>> [Total-Train] epoch: [%d], Accuracy: %.6f, Loss: %.6f, time: %.2f'
                      % (epoch, tot_train_acc, tot_train_loss, tot_train_time))
                # print('>> [Total-Test] epoch: [%d], Accuracy: %.6f, Loss: %.6f'
                #     % (epoch, tot_test_acc, tot_test_loss))

                # self.db.mon_data_to_db(epoch, tot_train_acc, tot_test_acc, tot_train_loss, tot_test_loss, tot_train_time, tot_test_time)

                # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'right/output_layer/logit/W_conv2d')[0]
                # print(sess.run(kernel))

                ## Save model
                os.makedirs(os.path.join(flags.FLAGS.trained_param_path), exist_ok=True)
                self._saver.save(sess, os.path.join(flags.FLAGS.trained_param_path, 'dementia_param'), global_step=epoch)
                print('>> [Model saved] epoch: %d' % (epoch))

            coord.request_stop()
            coord.join(threads)

            # self.db.close_conn()

    def cam_test(self, sort='grad_cam'):
        sample_num = 3  # 클래스 당 테스트 샘플0 개수
        class_num = 2  # 전체 클래스 개수
        batch_size = sample_num * class_num
        img_size = (224, 400)
        sample_path = '/home/kyh/dataset/gelontoxon_cam'

        def save_matplot_img(cam_outputs, sample_num, class_num, file_name):
            f = plt.figure(figsize=(10, 8))
            plt.suptitle('CAM (Class Activation Mapping)', fontsize=20)
            outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)

            inner = gridspec.GridSpecFromSubplotSpec(class_num, sample_num, subplot_spec=outer[0], wspace=0.1, hspace=0.1)

            for cls in range(class_num):
                for sample in range(sample_num):
                    subplot = plt.Subplot(f, inner[sample + cls * sample_num])
                    subplot.axis('off')
                    subplot.imshow(cam_outputs[sample + cls * sample_num])
                    f.add_subplot(subplot)

            f.savefig(os.path.join('/home/kyh/PycharmProjects/PythonRepository/Projects/DeepLearningTechniques/MobileNet_v2/cam_image', file_name))
            print('>> CAM Complete')

        def get_file_names():
            filename_list, label_list = [], []

            for cls in range(class_num):
                file_names = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(os.path.join(sample_path, str(cls)))]
                file_names = np.asarray(file_names).flatten()

                random_sort = np.random.permutation(file_names.shape[0])
                file_names = file_names[random_sort][:sample_num]

                for f_name in file_names:
                    filename_list.append(f_name)
                    label_list.append(cls)

            filename_list = np.asarray(filename_list)
            label_list = np.asarray(label_list)

            filename_tensor = tf.convert_to_tensor(filename_list, dtype=tf.string)
            label_tensor = tf.convert_to_tensor(label_list, dtype=tf.int32)

            return filename_tensor, label_tensor

        def normal_data(x, y):
            with tf.variable_scope('normal_data'):
                data = tf.read_file(x)
                data = tf.image.decode_png(data, channels=3, name='decode_img')
                data = tf.image.resize_images(data, size=img_size)
                data = tf.divide(data, 255.)
            return data, y

        def data_loader(filename_data, label_data):
            with tf.variable_scope('data_loader'):
                dataset = tf.contrib.data.Dataset.from_tensor_slices((filename_data, label_data)).repeat()

                normal_dataset_map = dataset.map(normal_data).batch(batch_size)
                normal_iterator = normal_dataset_map.make_one_shot_iterator()
                normal_batch_input = normal_iterator.get_next()

            return normal_batch_input

        filename_tensor, label_tensor = get_file_names()
        normal_batch_input = data_loader(filename_tensor, label_tensor)

        with tf.Session() as sess:
            filename_batch = sess.run(filename_tensor)
            data_batch = sess.run(normal_batch_input)

            model = Model(sess=sess, is_training=self.is_training, name='model')
            model.build()

            if sort == 'grad_cam':
                grad_cam = GradCAM(instance=model, sample_size=sample_num * class_num)
                grad_cam.build()
            else:
                guided_cam = GuidedGradCAM(instance=model, sample_size=sample_num * class_num)
                guided_cam.build()

            self._saver = tf.train.Saver()
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.trained_param_path))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            if sort == 'grad_cam':
                # 원본 영상과 Grad CAM 을 합친 결과
                grad_outputs = grad_cam.visualize(data_batch[0], filename_batch)
                save_matplot_img(grad_outputs, sample_num, class_num, 'grad_cam.png')
            else:
                guided_outputs = guided_cam.visualize(data_batch[0], filename_batch)
                save_matplot_img(guided_outputs, sample_num, class_num, 'guided_cam.png')

            '''텐서플로우 그래프 저장'''
            os.makedirs(os.path.join(flags.FLAGS.deploy_log_dir), exist_ok=True)
            tf.train.write_graph(sess.graph_def, flags.FLAGS.deploy_log_dir, 'graph.pbtxt')
            self._saver.save(sess, os.path.join(flags.FLAGS.deploy_log_dir, 'dementia_param'))

            '''PB File Save'''
            # builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(flags.FLAGS.deploy_log_dir, 'dementia_cam'))
            # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
            # builder.save()

neuralnet = Neuralnet(is_training=True)
# neuralnet.cam_test(sort='guided_cam')
# neuralnet.cam_test(sort='grad_cam')
neuralnet.train()