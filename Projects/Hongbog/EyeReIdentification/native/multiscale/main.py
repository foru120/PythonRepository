import time
from collections import Counter
import numpy as np
import os

from Projects.Hongbog.EyeReIdentification.native.multiscale.constants import *
from Projects.Hongbog.EyeReIdentification.native.multiscale.model import Model
from Projects.Hongbog.EyeReIdentification.native.multiscale.dataloader import DataLoader

class Neuralnet:

    def __init__(self):
        self.loader = DataLoader(batch_size=flags.FLAGS.batch_size,
                                 train_right_root_path=flags.FLAGS.right_train_data_path,
                                 test_right_root_path=flags.FLAGS.right_test_data_path,
                                 train_left_root_path=flags.FLAGS.left_train_data_path,
                                 test_left_root_path=flags.FLAGS.left_test_data_path)

    def train(self):
        self.loader.train_init()
        print('>> Train DataLoader created')

        train_num = self.loader.train_tot_len // flags.FLAGS.batch_size
        augmentation_cnt = 7

        train_right_low1, train_right_low2, train_right_low3, train_right_low4, train_right_low5, train_right_low6, train_right_low7, \
        train_left_low1, train_left_low2, train_left_low3, train_left_low4, train_left_low5, train_left_low6, train_left_low7 = self.loader.train_low_loader()
        train_right_mid1, train_right_mid2, train_right_mid3, train_right_mid4, train_right_mid5, train_right_mid6, train_right_mid7, \
        train_left_mid1, train_left_mid2, train_left_mid3, train_left_mid4, train_left_mid5, train_left_mid6, train_left_mid7 = self.loader.train_mid_loader()
        train_right_high1, train_right_high2, train_right_high3, train_right_high4, train_right_high5, train_right_high6, train_right_high7, \
        train_left_high1, train_left_high2, train_left_high3, train_left_high4, train_left_high5, train_left_high6, train_left_high7 = self.loader.train_high_loader()

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
        )

        with tf.Session(config=config) as sess:
            right_model = Model(sess=sess, lr=flags.FLAGS.learning_rate, batch_size=flags.FLAGS.batch_size, is_training=True, name='right')
            left_model = Model(sess=sess, lr=flags.FLAGS.learning_rate, batch_size=flags.FLAGS.batch_size, is_training=True, name='left')

            print('>> Tensorflow session built. Variables initialized')
            sess.run(tf.global_variables_initializer())

            '''훈련 데이터 및 텐서보드 모니터링 로그 저장 디렉토리 생성'''
            os.makedirs(flags.FLAGS.trained_weight_dir, exist_ok=True)
            os.makedirs(flags.FLAGS.save_weight_dir, exist_ok=True)

            '''텐서플로우 그래프 저장'''
            tf.train.write_graph(sess.graph_def, flags.FLAGS.trained_weight_dir, 'graph.pbtxt')
            print('>> Graph saved')

            self._saver = tf.train.Saver(var_list=tf.global_variables())
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.trained_weight_dir))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('>> Running started')

            for epoch in range(1, flags.FLAGS.epochs+1):
                tot_train_right_acc, tot_train_right_loss = [], []
                tot_train_left_acc, tot_train_left_loss = [], []

                if epoch % 5 == 0:
                    right_model.lr = max(right_model.lr / 2, 0.0001)
                    left_model.lr = max(left_model.lr / 2, 0.0001)

                '''Model Train'''
                train_st = time.time()
                for step in range(1, train_num+1):
                    '''Data Loading - low, middle, high 당 (right, left 300 개씩)'''
                    train_low_data = sess.run([train_right_low1, train_right_low2, train_right_low3, train_right_low4, train_right_low5, train_right_low6, train_right_low7,
                                               train_left_low1, train_left_low2, train_left_low3, train_left_low4, train_left_low5, train_left_low6, train_left_low7])

                    train_low_right_batch_ori_x, train_low_right_batch_query_x, train_low_right_batch_y = np.concatenate([right_data[0] for right_data in train_low_data[:augmentation_cnt]]), \
                                                                                                          np.concatenate([right_data[1] for right_data in train_low_data[:augmentation_cnt]]), \
                                                                                                          np.concatenate([right_data[2] for right_data in train_low_data[:augmentation_cnt]])
                    train_low_left_batch_ori_x, train_low_left_batch_query_x, train_low_left_batch_y = np.concatenate([left_data[0] for left_data in train_low_data[augmentation_cnt:]]), \
                                                                                                       np.concatenate([left_data[1] for left_data in train_low_data[augmentation_cnt:]]), \
                                                                                                       np.concatenate([left_data[2] for left_data in train_low_data[augmentation_cnt:]])

                    train_mid_data = sess.run([train_right_mid1, train_right_mid2, train_right_mid3, train_right_mid4, train_right_mid5, train_right_mid6, train_right_mid7,
                                               train_left_mid1, train_left_mid2, train_left_mid3, train_left_mid4, train_left_mid5, train_left_mid6, train_left_mid7])
                    train_mid_right_batch_ori_x, train_mid_right_batch_query_x, train_mid_right_batch_y = np.concatenate([right_data[0] for right_data in train_mid_data[:augmentation_cnt]]), \
                                                                                                          np.concatenate([right_data[1] for right_data in train_mid_data[:augmentation_cnt]]), \
                                                                                                          np.concatenate([right_data[2] for right_data in train_mid_data[:augmentation_cnt]])
                    train_mid_left_batch_ori_x, train_mid_left_batch_query_x, train_mid_left_batch_y = np.concatenate([left_data[0] for left_data in train_mid_data[augmentation_cnt:]]), \
                                                                                                       np.concatenate([left_data[1] for left_data in train_mid_data[augmentation_cnt:]]), \
                                                                                                       np.concatenate([left_data[2] for left_data in train_mid_data[augmentation_cnt:]])

                    train_high_data = sess.run([train_right_high1, train_right_high2, train_right_high3, train_right_high4, train_right_high5, train_right_high6, train_right_high7,
                                                train_left_high1, train_left_high2, train_left_high3, train_left_high4, train_left_high5, train_left_high6, train_left_high7])
                    train_high_right_batch_ori_x, train_high_right_batch_query_x, train_high_right_batch_y = np.concatenate([right_data[0] for right_data in train_high_data[:augmentation_cnt]]), \
                                                                                                             np.concatenate([right_data[1] for right_data in train_high_data[:augmentation_cnt]]), \
                                                                                                             np.concatenate([right_data[2] for right_data in train_high_data[:augmentation_cnt]])
                    train_high_left_batch_ori_x, train_high_left_batch_query_x, train_high_left_batch_y = np.concatenate([left_data[0] for left_data in train_high_data[augmentation_cnt:]]), \
                                                                                                          np.concatenate([left_data[1] for left_data in train_high_data[augmentation_cnt:]]), \
                                                                                                          np.concatenate([left_data[1] for left_data in train_high_data[augmentation_cnt:]])

                    st = time.time()
                    step_train_right_acc, step_train_right_loss = [], []
                    for idx in range(0, flags.FLAGS.batch_size*augmentation_cnt, flags.FLAGS.batch_size):
                        train_right_acc, train_right_loss, _ = right_model.train(ori_low_res_X=train_low_right_batch_ori_x[idx:idx+flags.FLAGS.batch_size],
                                                                                 ori_mid_res_X=train_mid_right_batch_ori_x[idx:idx+flags.FLAGS.batch_size],
                                                                                 ori_high_res_X=train_high_left_batch_ori_x[idx:idx+flags.FLAGS.batch_size],
                                                                                 query_low_res_X=train_low_right_batch_query_x[idx:idx+flags.FLAGS.batch_size],
                                                                                 query_mid_res_X=train_mid_right_batch_query_x[idx:idx+flags.FLAGS.batch_size],
                                                                                 query_high_res_X=train_high_right_batch_query_x[idx:idx+flags.FLAGS.batch_size],
                                                                                 y=train_low_right_batch_y[idx:idx+flags.FLAGS.batch_size])
                        step_train_right_acc.append(train_right_acc)
                        step_train_right_loss.append(train_right_loss)
                        tot_train_right_acc.append(train_right_acc)
                        tot_train_right_loss.append(train_right_loss)

                    step_train_left_acc, step_train_left_loss = [], []
                    for idx in range(0, flags.FLAGS.batch_size*augmentation_cnt, flags.FLAGS.batch_size):
                        train_left_acc, train_left_loss, _ = left_model.train(ori_low_res_X=train_low_left_batch_ori_x[idx:idx+flags.FLAGS.batch_size],
                                                                              ori_mid_res_X=train_mid_left_batch_ori_x[idx:idx+flags.FLAGS.batch_size],
                                                                              ori_high_res_X=train_high_left_batch_ori_x[idx:idx+flags.FLAGS.batch_size],
                                                                              query_low_res_X=train_low_left_batch_query_x[idx:idx+flags.FLAGS.batch_size],
                                                                              query_mid_res_X=train_mid_left_batch_query_x[idx:idx+flags.FLAGS.batch_size],
                                                                              query_high_res_X=train_high_left_batch_query_x[idx:idx+flags.FLAGS.batch_size],
                                                                              y=train_low_left_batch_y[idx:idx + flags.FLAGS.batch_size])

                        step_train_left_acc.append(train_left_acc)
                        step_train_left_loss.append(train_left_loss)
                        tot_train_left_acc.append(train_left_acc)
                        tot_train_left_loss.append(train_left_loss)
                    et = time.time()

                    step_train_right_acc = float(np.mean(np.array(step_train_right_acc)))
                    step_train_right_loss = float(np.mean(np.array(step_train_right_loss)))
                    step_train_left_acc = float(np.mean(np.array(step_train_left_acc)))
                    step_train_left_loss = float(np.mean(np.array(step_train_left_loss)))
                    print(">> [Step-Train] epoch/step: [%d/%d], [Right]Accuracy: %.6f, [Left]Accuracy: %.6f, [Right]Loss: %.6f, [Left]Loss: %.6f, step_time: %.2f"
                          % (epoch, step, step_train_right_acc, step_train_left_acc, step_train_right_loss, step_train_left_loss, et - st))

                train_et = time.time()
                tot_train_time = train_et - train_st

                tot_train_right_acc = float(np.mean(np.array(tot_train_right_acc)))
                tot_train_right_loss = float(np.mean(np.array(tot_train_right_loss)))
                tot_train_left_acc = float(np.mean(np.array(tot_train_left_acc)))
                tot_train_left_loss = float(np.mean(np.array(tot_train_left_loss)))

                print('>> [Total-Train] epoch: [%d], [Right]Accuracy: %.6f, [Left]Accuracy: %.6f, [Right]Loss: %.6f, [Left]Loss: %.6f, time: %.2f'
                      % (epoch, tot_train_right_acc, tot_train_left_acc, tot_train_right_loss, tot_train_left_loss, tot_train_time))

                '''Database 에 로그 저장'''
                # self.db.mon_data_to_db(epoch, tot_train_acc, tot_test_acc, tot_train_loss, tot_test_loss, tot_train_time, tot_test_time)

                '''특정 레이어의 변수 값 출력'''
                # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'right/output_layer/logit/W_conv2d')[0]
                # print(sess.run(kernel))

                '''CKPT, parameter File Save'''
                self._saver.save(sess, os.path.join(flags.FLAGS.save_weight_dir, 'eye_re_identification'), global_step=epoch)

                ## PB File Save
                # builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(flags.FLAGS.trained_weight_dir, 'eye_verification_param'))
                # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
                # builder.save()
                print('>> [Model saved] epoch: %d' % (epoch))

            coord.request_stop()
            coord.join(threads)

            # self.db.close_conn()

    def integration_test(self):
        tf.reset_default_graph()

        self.loader.test_init()
        print('>> Test DataLoader created')

        test_num = self.loader.test_tot_len // flags.FLAGS.batch_size

        test_right_low, test_left_low = self.loader.test_low_loader()
        test_right_mid, test_left_mid = self.loader.test_mid_loader()
        test_right_high, test_left_high = self.loader.test_high_loader()

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
        )

        with tf.Session(config=config) as sess:
            right_model = Model(sess=sess, lr=flags.FLAGS.learning_rate, batch_size=flags.FLAGS.batch_size, is_training=False, name='right')
            left_model = Model(sess=sess, lr=flags.FLAGS.learning_rate, batch_size=flags.FLAGS.batch_size, is_training=False, name='left')

            self._saver = tf.train.Saver(var_list=tf.global_variables())
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.trained_weight_dir))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('>> Running started')

            '''Model Test'''
            tot_test_right_acc, tot_test_right_loss = [], []
            tot_test_left_acc, tot_test_left_loss = [], []

            right_prob, right_label = [], []
            left_prob, left_label = [], []
            ensemble_prob, ensemble_label = [], []
            tot_ensemble_acc = []

            for step in range(1, test_num + 1):
                test_low_data = sess.run([test_right_low, test_left_low])
                test_low_right_batch_ori_x, test_low_right_batch_query_x, test_low_right_batch_y, test_low_left_batch_ori_x, test_low_left_batch_query_x, test_low_left_batch_y = \
                    test_low_data[0][0], test_low_data[0][1], test_low_data[0][2], test_low_data[1][0], test_low_data[1][1], test_low_data[1][2]

                test_mid_data = sess.run([test_right_mid, test_left_mid])
                test_mid_right_batch_ori_x, test_mid_right_batch_query_x, test_mid_right_batch_y, test_mid_left_batch_ori_x, test_mid_left_batch_query_x, test_mid_left_batch_y = \
                    test_mid_data[0][0], test_mid_data[0][1], test_mid_data[0][2], test_mid_data[1][0], test_mid_data[1][1], test_mid_data[1][2]

                test_high_data = sess.run([test_right_high, test_left_high])
                test_high_right_batch_ori_x, test_high_right_batch_query_x, test_high_right_batch_y, test_high_left_batch_ori_x, test_high_left_batch_query_x, test_high_left_batch_y = \
                    test_high_data[0][0], test_high_data[0][1], test_high_data[0][2], test_high_data[1][0], test_high_data[1][1], test_high_data[1][2]

                test_right_acc, test_right_loss, test_right_prob = right_model.validation(
                    ori_low_res_X=test_low_right_batch_ori_x,
                    ori_mid_res_X=test_mid_right_batch_ori_x,
                    ori_high_res_X=test_high_right_batch_ori_x,
                    query_low_res_X=test_low_right_batch_query_x,
                    query_mid_res_X=test_mid_right_batch_query_x,
                    query_high_res_X=test_high_right_batch_query_x,
                    y=test_low_right_batch_y)

                test_left_acc, test_left_loss, test_left_prob = left_model.validation(
                    ori_low_res_X=test_low_left_batch_ori_x,
                    ori_mid_res_X=test_mid_right_batch_ori_x,
                    ori_high_res_X=test_high_right_batch_ori_x,
                    query_low_res_X=test_low_left_batch_query_x,
                    query_mid_res_X=test_mid_left_batch_query_x,
                    query_high_res_X=test_high_left_batch_query_x,
                    y=test_low_left_batch_y)

                '''Ensemble Prediction'''
                tot_ensemble_acc.append(np.sum(np.argmax(test_right_prob + test_left_prob, axis=1) == np.array(
                    test_low_right_batch_y)) / flags.FLAGS.batch_size)

                '''Monitoring'''
                tot_test_right_acc.append(test_right_acc)
                tot_test_right_loss.append(test_right_loss)
                tot_test_left_acc.append(test_left_acc)
                tot_test_left_loss.append(test_left_loss)

                '''Confusion Matrix'''
                right_prob.append(np.argmax(test_right_prob, axis=1).flatten().tolist())
                right_label.append(test_low_right_batch_y.flatten().tolist())
                left_prob.append(np.argmax(test_left_prob, axis=1).flatten().tolist())
                left_label.append(test_low_left_batch_y.flatten().tolist())
                ensemble_prob.append(np.argmax(test_right_prob + test_left_prob, axis=1).flatten().tolist())
                ensemble_label.append(test_low_right_batch_y.flatten().tolist())

            tot_test_right_acc = float(np.mean(np.array(tot_test_right_acc)))
            tot_test_right_loss = float(np.mean(np.array(tot_test_right_loss)))
            tot_test_left_acc = float(np.mean(np.array(tot_test_left_acc)))
            tot_test_left_loss = float(np.mean(np.array(tot_test_left_loss)))
            tot_ensemble_acc = float(np.mean(np.array(tot_ensemble_acc)))

            print('>> [Total-Test] [Right]Accuracy: %.6f, [Left]Accuracy: %.6f, [Ensemble]Accuracy: %.6f, [Right]Loss: %.6f, [Left]Loss: %.6f'
                % (tot_test_right_acc, tot_test_left_acc, tot_ensemble_acc, tot_test_right_loss, tot_test_left_loss))
            print('>> [Right-Confusion-Matrix]')
            print(sess.run(tf.confusion_matrix(labels=np.array(right_label).flatten(), predictions=np.array(right_prob).flatten(), num_classes=2)))
            print('>> [Left-Confusion-Matrix]')
            print(sess.run(tf.confusion_matrix(labels=np.array(left_label).flatten(), predictions=np.array(left_prob).flatten(), num_classes=2)))
            print('>> [Ensemble-Confusion-Matrix]')
            print(sess.run(tf.confusion_matrix(labels=np.array(ensemble_label).flatten(), predictions=np.array(ensemble_prob).flatten(), num_classes=2)))

            coord.request_stop()
            coord.join(threads)

    def unit_test(self):
        low_img_size, mid_img_size, high_img_size = (46, 100), (70, 150), (92, 200)
        ori_right_path = 'G:/04_dataset/re_identification/eye_ori/unit_test/right/ori'
        query_right_path = 'G:/04_dataset/re_identification/eye_ori/unit_test/right/query'
        ori_left_path = 'G:/04_dataset/re_identification/eye_ori/unit_test/left/ori'
        query_left_path = 'G:/04_dataset/re_identification/eye_ori/unit_test/left/query'

        def get_file_names():
            right_pair_file_names, left_pair_file_names = [], []

            ori_right_file_names = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(ori_right_path)]
            ori_right_file_names = np.array(ori_right_file_names).flatten()
            query_right_file_names = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(query_right_path)]
            query_right_file_names = np.array(query_right_file_names).flatten()

            ori_left_file_names = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(ori_left_path)]
            ori_left_file_names = np.array(ori_left_file_names).flatten()
            query_left_file_names = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(query_left_path)]
            query_left_file_names = np.array(query_left_file_names).flatten()

            for ori_right_file in ori_right_file_names:
                for query_right_file in query_right_file_names:
                    right_pair_file_names.append((ori_right_file, query_right_file))

            for ori_left_file in ori_left_file_names:
                for query_left_file in query_left_file_names:
                    left_pair_file_names.append((ori_left_file, query_left_file))

            for pair_name in right_pair_file_names:
                print(pair_name)
            for pair_name in left_pair_file_names:
                print(pair_name)

            right_pair_file_names = tf.convert_to_tensor(right_pair_file_names, dtype=tf.string)
            left_pair_file_names = tf.convert_to_tensor(left_pair_file_names, dtype=tf.string)

            return right_pair_file_names, left_pair_file_names

        def tf_equalize_histogram(image):
            values_range = tf.constant([0., 255.], dtype=tf.float32)
            histogram = tf.histogram_fixed_width(tf.to_float(image), values_range, 256)
            cdf = tf.cumsum(histogram)
            cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

            img_shape = tf.shape(image)
            pix_cnt = img_shape[-3] * img_shape[-2]
            px_map = tf.round(tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt - 1))
            px_map = tf.cast(px_map, tf.uint8)

            eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
            return eq_hist

        def low_normal_data(x):
            with tf.variable_scope('low_normal_data'):
                ori_x = tf.read_file(x[0])
                ori_x = tf.image.decode_png(ori_x, channels=1)
                ori_x = tf.image.resize_images(ori_x, size=low_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                ori_x = tf_equalize_histogram(ori_x)
                ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

                query_x = tf.read_file(x[1])
                query_x = tf.image.decode_png(query_x, channels=1)
                query_x = tf.image.resize_images(query_x, size=low_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                query_x = tf_equalize_histogram(query_x)
                query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
            return ori_x, query_x

        def mid_normal_data(x):
            with tf.variable_scope('mid_normal_data'):
                ori_x = tf.read_file(x[0])
                ori_x = tf.image.decode_png(ori_x, channels=1)
                ori_x = tf.image.resize_images(ori_x, size=mid_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                ori_x = tf_equalize_histogram(ori_x)
                ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

                query_x = tf.read_file(x[1])
                query_x = tf.image.decode_png(query_x, channels=1)
                query_x = tf.image.resize_images(query_x, size=mid_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                query_x = tf_equalize_histogram(query_x)
                query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
            return ori_x, query_x

        def high_normal_data(x):
            with tf.variable_scope('high_normal_data'):
                ori_x = tf.read_file(x[0])
                ori_x = tf.image.decode_png(ori_x, channels=1)
                ori_x = tf.image.resize_images(ori_x, size=high_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                ori_x = tf_equalize_histogram(ori_x)
                ori_x = tf.divide(tf.cast(ori_x, tf.float32), 255.)

                query_x = tf.read_file(x[1])
                query_x = tf.image.decode_png(query_x, channels=1)
                query_x = tf.image.resize_images(query_x, size=high_img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                query_x = tf_equalize_histogram(query_x)
                query_x = tf.divide(tf.cast(query_x, tf.float32), 255.)
            return ori_x, query_x

        def data_loader(right_file_names, left_file_names):
            with tf.variable_scope('data_loader'):
                # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
                right_dataset = tf.contrib.data.Dataset.from_tensor_slices(right_file_names).repeat()
                left_dataset = tf.contrib.data.Dataset.from_tensor_slices(left_file_names).repeat()

                right_low_dataset_map = right_dataset.map(low_normal_data).batch(flags.FLAGS.batch_size)
                right_low_iterator = right_low_dataset_map.make_one_shot_iterator()
                right_low_batch_input = right_low_iterator.get_next()

                left_low_dataset_map = left_dataset.map(low_normal_data).batch(flags.FLAGS.batch_size)
                left_low_iterator = left_low_dataset_map.make_one_shot_iterator()
                left_low_batch_input = left_low_iterator.get_next()

                right_mid_dataset_map = right_dataset.map(mid_normal_data).batch(flags.FLAGS.batch_size)
                right_mid_iterator = right_mid_dataset_map.make_one_shot_iterator()
                right_mid_batch_input = right_mid_iterator.get_next()

                left_mid_dataset_map = left_dataset.map(mid_normal_data).batch(flags.FLAGS.batch_size)
                left_mid_iterator = left_mid_dataset_map.make_one_shot_iterator()
                left_mid_batch_input = left_mid_iterator.get_next()

                right_high_dataset_map = right_dataset.map(high_normal_data).batch(flags.FLAGS.batch_size)
                right_high_iterator = right_high_dataset_map.make_one_shot_iterator()
                right_high_batch_input = right_high_iterator.get_next()

                left_high_dataset_map = left_dataset.map(high_normal_data).batch(flags.FLAGS.batch_size)
                left_high_iterator = left_high_dataset_map.make_one_shot_iterator()
                left_high_batch_input = left_high_iterator.get_next()

            return right_low_batch_input, left_low_batch_input, right_mid_batch_input, left_mid_batch_input, right_high_batch_input, left_high_batch_input

        right_pair_file_names, left_pair_file_names = get_file_names()
        low_right_batch_img, low_left_batch_img, mid_right_batch_img, mid_left_batch_img, high_right_batch_img, high_left_batch_img = data_loader(right_pair_file_names, left_pair_file_names)

        with tf.Session() as sess:
            right_model = Model(sess=sess, lr=flags.FLAGS.learning_rate, batch_size=flags.FLAGS.batch_size, is_training=False, name='right')
            left_model = Model(sess=sess, lr=flags.FLAGS.learning_rate, batch_size=flags.FLAGS.batch_size, is_training=False, name='left')

            self._saver = tf.train.Saver(var_list=tf.global_variables())
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.trained_weight_dir))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                print(ckpt_st.model_checkpoint_path)
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'right/output_layer/logit/W_conv2d')[0]
            # print(sess.run(kernel))

            ensemble_pred = []
            ensemble_prob = []

            for _ in range(0, right_pair_file_names.shape[0], flags.FLAGS.batch_size):
                low_right_batch_x, low_left_batch_x, mid_right_batch_x, mid_left_batch_x, high_right_batch_x, high_left_batch_x = \
                    sess.run([low_right_batch_img, low_left_batch_img, mid_right_batch_img, mid_left_batch_img, high_right_batch_img, high_left_batch_img])
                right_prob = right_model.predict(ori_low_res_X=low_right_batch_x[0],
                                                 ori_mid_res_X=mid_right_batch_x[0],
                                                 ori_high_res_X=high_right_batch_x[0],
                                                 query_low_res_X=low_right_batch_x[1],
                                                 query_mid_res_X=mid_right_batch_x[1],
                                                 query_high_res_X=high_right_batch_x[1])
                left_prob = left_model.predict(ori_low_res_X=low_left_batch_x[0],
                                               ori_mid_res_X=mid_left_batch_x[0],
                                               ori_high_res_X=high_left_batch_x[0],
                                               query_low_res_X=low_left_batch_x[1],
                                               query_mid_res_X=mid_left_batch_x[1],
                                               query_high_res_X=high_left_batch_x[1])
                ensemble_prob.append(np.max(right_prob + left_prob, axis=1).tolist())
                ensemble_pred.append(np.argmax(right_prob + left_prob, axis=1).tolist())

            '''Ensemble Prediction'''
            ensemble_pred = np.array(ensemble_pred).flatten()
            for pred in ensemble_pred:
                print(pred, end=',')
            print('')
            ensemble_pred = Counter(ensemble_pred)
            print(ensemble_pred.most_common(1)[0][0])

            for prob in ensemble_prob:
                print(prob)

            ensemble_prob = np.asarray(ensemble_prob)
            low_acc = ensemble_prob[ensemble_prob < 1.99]
            print(np.sort(low_acc))
            print(len(low_acc))

            os.makedirs(flags.FLAGS.deploy_log_dir, exist_ok=True)

            '''Graph Save'''
            tf.train.write_graph(sess.graph_def, flags.FLAGS.deploy_log_dir, 'graph.pbtxt')
            self._saver.save(sess, os.path.join(flags.FLAGS.deploy_log_dir, 'model_graph'))
            print('>> Graph saved')

            '''PB File Save'''
            # builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(flags.FLAGS.deploy_log_dir, 'eye_verification_param'))
            # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
            # builder.save()

neuralnet = Neuralnet()
# neuralnet.cam_test()
neuralnet.train()
# neuralnet.integration_test()
# neuralnet.unit_test()