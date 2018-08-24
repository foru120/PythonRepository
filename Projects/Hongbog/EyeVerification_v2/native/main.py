import time
from collections import Counter
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Projects.Hongbog.EyeVerification_v2.native.constant import *
from Projects.Hongbog.EyeVerification_v2.native.model import Model
from Projects.Hongbog.EyeVerification_v2.native.dataloader import DataLoader
from Projects.Hongbog.EyeVerification_v2.native.cam import GradCAM

class Neuralnet:

    def __init__(self):
        self.loader = DataLoader(batch_size=flags.FLAGS.batch_size,
                                 train_root_path=flags.FLAGS.train_data_path,
                                 test_root_path=flags.FLAGS.test_data_path)

    def train(self):
        self.loader.train_init()
        print('>> Train DataLoader created')

        train_num = self.loader.train_tot_len // flags.FLAGS.batch_size

        train_low1, train_low2, train_low3, train_low4, train_low5, train_low6 = self.loader.train_low_loader()
        train_mid1, train_mid2, train_mid3, train_mid4, train_mid5, train_mid6 = self.loader.train_mid_loader()
        train_high1, train_high2, train_high3, train_high4, train_high5, train_high6 = self.loader.train_high_loader()

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
        )

        with tf.Session(config=config) as sess:
            model = Model(sess=sess, lr=flags.FLAGS.learning_rate, is_training=True, name='pair')

            print('>> Tensorflow session built. Variables initialized')
            sess.run(tf.global_variables_initializer())

            '''훈련 데이터 및 텐서보드 모니터링 로그 저장 디렉토리 생성'''
            os.makedirs(flags.FLAGS.trained_weight_dir, exist_ok=True)
            os.makedirs(os.path.join(flags.FLAGS.tensorboard_log_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(flags.FLAGS.tensorboard_log_dir, 'test'), exist_ok=True)

            '''텐서플로우 그래프 저장'''
            tf.train.write_graph(sess.graph_def, flags.FLAGS.trained_weight_dir, 'graph.pbtxt')
            print('>> Graph saved')

            self._saver = tf.train.Saver(var_list=tf.global_variables())
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.trained_weight_dir))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                #self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('>> Running started')

            for epoch in range(1, flags.FLAGS.epochs+1):
                tot_train_acc, tot_train_loss = [], []

                if epoch % 5 == 0:
                    model.lr = max(model.lr / 2, 0.0001)

                '''Model Train'''
                train_st = time.time()
                for step in range(1, train_num+1):
                    '''Data Loading - low, middle, high 당 (right, left 300 개씩)'''
                    train_low_data = sess.run([train_low1, train_low2, train_low3, train_low4, train_low5, train_low6])
                    train_low_batch_x, train_low_batch_y = np.concatenate([data[0] for data in train_low_data]), np.concatenate([data[1] for data in train_low_data])

                    train_mid_data = sess.run([train_mid1, train_mid2, train_mid3, train_mid4, train_mid5, train_mid6])
                    train_mid_batch_x, train_mid_batch_y = np.concatenate([data[0] for data in train_mid_data]), np.concatenate([data[1] for data in train_mid_data])

                    train_high_data = sess.run([train_high1, train_high2, train_high3, train_high4, train_high5, train_high6])
                    train_high_batch_x, train_high_batch_y = np.concatenate([data[0] for data in train_high_data]), np.concatenate([data[1] for data in train_high_data])

                    st = time.time()
                    step_train_acc, step_train_loss = [], []
                    for idx in range(0, flags.FLAGS.batch_size*6, flags.FLAGS.batch_size):
                        train_acc, train_loss, _ = model.train(low_res_X=train_low_batch_x[idx:idx + flags.FLAGS.batch_size],
                                                               mid_res_X=train_mid_batch_x[idx:idx + flags.FLAGS.batch_size],
                                                               high_res_X=train_high_batch_x[idx:idx + flags.FLAGS.batch_size],
                                                               y=train_low_batch_y[idx:idx + flags.FLAGS.batch_size])

                        step_train_acc.append(train_acc)
                        step_train_loss.append(train_loss)
                        tot_train_acc.append(train_acc)
                        tot_train_loss.append(train_loss)

                    et = time.time()

                    step_train_acc = float(np.mean(np.array(step_train_acc)))
                    step_train_loss = float(np.mean(np.array(step_train_loss)))
                    print(">> [Step-Train] epoch/step: [%d/%d], Accuracy: %.6f, Loss: %.6f, step_time: %.2f" % (epoch, step, step_train_acc, step_train_loss, et - st))

                train_et = time.time()
                tot_train_time = train_et - train_st

                tot_train_acc = float(np.mean(np.array(tot_train_acc)))
                tot_train_loss = float(np.mean(np.array(tot_train_loss)))

                print('>> [Total-Train] epoch: [%d], Accuracy: %.6f, Loss: %.6f, time: %.2f' % (epoch, tot_train_acc, tot_train_loss, tot_train_time))

                '''Database 에 로그 저장'''
                # self.db.mon_data_to_db(epoch, tot_train_acc, tot_test_acc, tot_train_loss, tot_test_loss, tot_train_time, tot_test_time)

                '''특정 레이어의 변수 값 출력'''
                # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'right/output_layer/logit/W_conv2d')[0]
                # print(sess.run(kernel))

                '''CKPT, parameter File Save'''
                self._saver.save(sess, os.path.join(flags.FLAGS.trained_weight_dir, 'eye_verification_param'), global_step=epoch)

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

        test_low = self.loader.test_low_loader()
        test_mid = self.loader.test_mid_loader()
        test_high = self.loader.test_high_loader()

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
        )

        with tf.Session(config=config) as sess:
            model = Model(sess=sess, lr=flags.FLAGS.learning_rate, is_training=False, name='pair')

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
            tot_test_acc, tot_test_loss = [], []

            prob, label = [], []

            for step in range(1, test_num + 1):
                test_low_data = sess.run(test_low)
                test_low_batch_x, test_low_batch_y = test_low_data[0], test_low_data[1]

                test_mid_data = sess.run(test_mid)
                test_mid_batch_x, test_mid_batch_y = test_mid_data[0], test_mid_data[1]

                test_high_data = sess.run(test_high)
                test_high_batch_x, test_high_batch_y = test_high_data[0], test_high_data[1]

                test_acc, test_loss, test_prob = model.validation(
                    low_res_X=test_low_batch_x,
                    mid_res_X=test_mid_batch_x,
                    high_res_X=test_high_batch_x,
                    y=test_low_batch_y
                )

                '''Monitoring'''
                tot_test_acc.append(test_acc)
                tot_test_loss.append(test_loss)

                '''Confusion Matrix'''
                prob.append(np.argmax(test_prob, axis=1).flatten().tolist())
                label.append(test_low_batch_y.flatten().tolist())

            tot_test_acc = float(np.mean(np.array(tot_test_acc)))
            tot_test_loss = float(np.mean(np.array(tot_test_loss)))

            print('>> [Total-Test] Accuracy: %.6f, Loss: %.6f' % (tot_test_acc, tot_test_loss))
            print('>> [Right-Confusion-Matrix]')
            print(sess.run(tf.confusion_matrix(labels=np.array(label).flatten(), predictions=np.array(prob).flatten(), num_classes=7)))

            coord.request_stop()
            coord.join(threads)

    def cam_test(self):
        sample_num = 3  # 클래스 당 테스트 샘플 개수
        class_num = 7  # 전체 클래스 개수
        batch_size = sample_num * class_num
        low_img_size, mid_img_size, high_img_size = (60, 160), (80, 200), (100, 240)
        sample_path = 'G:/04_dataset/eye_verification/pair_eye/test'

        def save_matplot_img(outputs, sample_num, class_num):
            f = plt.figure(figsize=(10, 8))
            plt.suptitle('Grad CAM (Gradient-weighted Class Activation Mapping)', fontsize=20)
            outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

            inner = gridspec.GridSpecFromSubplotSpec(class_num, sample_num, subplot_spec=outer[0], wspace=0.1, hspace=0.1)

            for cls in range(class_num):
                for sample in range(sample_num):
                    subplot = plt.Subplot(f, inner[sample + cls * sample_num])
                    subplot.axis('off')
                    subplot.imshow(outputs[sample + cls * sample_num])
                    f.add_subplot(subplot)

            f.savefig('D:/Source/PythonRepository/Projects/Hongbog/EyeReIdentification/native/cam_log/cam_test.png')
            print('>> Grad CAM Complete')

        def get_file_names():
            file_names = []

            for cls in range(class_num):
                file_name = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(os.path.join(sample_path, str(cls)))]
                file_name = np.array(file_name).flatten()

                random_sort = np.random.permutation(file_name.shape[0])
                file_name = file_name[random_sort][:sample_num]

                for f_name in file_name:
                    file_names.append(f_name)

            file_names = tf.convert_to_tensor(file_names, dtype=tf.string)

            return file_names

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

        def low_normal_data(path):
            with tf.variable_scope('low_normal_data'):
                data = tf.read_file(path)
                data = tf.image.decode_png(data, channels=1, name='decode_img')
                data = tf.image.resize_images(data, size=low_img_size)
                data = tf_equalize_histogram(data)
                data = tf.divide(data, 255.)
            return data

        def mid_normal_data(path):
            with tf.variable_scope('mid_normal_data'):
                data = tf.read_file(path)
                data = tf.image.decode_png(data, channels=1, name='decode_img')
                data = tf.image.resize_images(data, size=mid_img_size)
                data = tf_equalize_histogram(data)
                data = tf.divide(data, 255.)
            return data

        def high_normal_data(path):
            with tf.variable_scope('high_normal_data'):
                data = tf.read_file(path)
                data = tf.image.decode_png(data, channels=1, name='decode_img')
                data = tf.image.resize_images(data, size=high_img_size)
                data = tf_equalize_histogram(data)
                data = tf.divide(data, 255.)
            return data

        def data_loader(file_names):
            with tf.variable_scope('data_loader'):
                # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
                dataset = tf.contrib.data.Dataset.from_tensor_slices(file_names).repeat()

                low_dataset_map = dataset.map(low_normal_data).batch(batch_size)
                low_iterator = low_dataset_map.make_one_shot_iterator()
                low_batch_input = low_iterator.get_next()

                mid_dataset_map = dataset.map(mid_normal_data).batch(batch_size)
                mid_iterator = mid_dataset_map.make_one_shot_iterator()
                mid_batch_input = mid_iterator.get_next()

                high_dataset_map = dataset.map(high_normal_data).batch(batch_size)
                high_iterator = high_dataset_map.make_one_shot_iterator()
                high_batch_input = high_iterator.get_next()

            return low_batch_input, mid_batch_input, high_batch_input

        file_names = get_file_names()
        low_batch_img, mid_batch_img, high_batch_img = data_loader(file_names)

        with tf.Session() as sess:
            model = Model(sess=sess, lr=flags.FLAGS.learning_rate, is_training=False, name='pair')

            cam = GradCAM(instance=model, sample_size=sample_num * class_num, name='grad_cam')
            cam.build()

            self._saver = tf.train.Saver()
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.trained_weight_dir))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            low_batch_x, mid_batch_x, high_batch_x = sess.run([low_batch_img, mid_batch_img, high_batch_img])
            file_names = sess.run(file_names)

            cam_outputs = cam.visualize(low_res_X=low_batch_x,
                                        mid_res_X=mid_batch_x,
                                        high_res_X=high_batch_x,
                                        file_names=file_names)

            save_matplot_img(cam_outputs, sample_num, class_num)

    def unit_test(self):
        low_img_size, mid_img_size, high_img_size = (60, 160), (80, 200), (100, 240)
        sample_path = 'G:/04_dataset/eye_verification/pair_eye/test/6'

        def get_file_names():
            file_names = []

            for cls in range(class_num):
                file_name = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(os.path.join(sample_path, str(cls)))]
                file_name = np.array(file_name).flatten()

                random_sort = np.random.permutation(file_name.shape[0])
                file_name = file_name[random_sort][:sample_num]

                for f_name in file_name:
                    file_names.append(f_name)

            file_names = tf.convert_to_tensor(file_names, dtype=tf.string)

            return file_names

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

        def low_normal_data(path):
            with tf.variable_scope('low_normal_data'):
                data = tf.read_file(path)
                data = tf.image.decode_png(data, channels=1, name='decode_img')
                data = tf.image.resize_images(data, size=low_img_size)
                data = tf_equalize_histogram(data)
                data = tf.divide(data, 255.)
            return data

        def mid_normal_data(path):
            with tf.variable_scope('mid_normal_data'):
                data = tf.read_file(path)
                data = tf.image.decode_png(data, channels=1, name='decode_img')
                data = tf.image.resize_images(data, size=mid_img_size)
                data = tf_equalize_histogram(data)
                data = tf.divide(data, 255.)
            return data

        def high_normal_data(path):
            with tf.variable_scope('high_normal_data'):
                data = tf.read_file(path)
                data = tf.image.decode_png(data, channels=1, name='decode_img')
                data = tf.image.resize_images(data, size=high_img_size)
                data = tf_equalize_histogram(data)
                data = tf.divide(data, 255.)
            return data

        def data_loader(file_names):
            with tf.variable_scope('data_loader'):
                # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
                dataset = tf.contrib.data.Dataset.from_tensor_slices(file_names).repeat()

                low_dataset_map = dataset.map(low_normal_data).batch(batch_size)
                low_iterator = low_dataset_map.make_one_shot_iterator()
                low_batch_input = low_iterator.get_next()

                mid_dataset_map = dataset.map(mid_normal_data).batch(batch_size)
                mid_iterator = mid_dataset_map.make_one_shot_iterator()
                mid_batch_input = mid_iterator.get_next()

                high_dataset_map = dataset.map(high_normal_data).batch(batch_size)
                high_iterator = high_dataset_map.make_one_shot_iterator()
                high_batch_input = high_iterator.get_next()

            return low_batch_input, mid_batch_input, high_batch_input

        file_names = get_file_names()
        low_batch_img, mid_batch_img, high_batch_img = data_loader(file_names)

        with tf.Session() as sess:
            model = Model(sess=sess, lr=flags.FLAGS.learning_rate, is_training=False, name='pair')

            self._saver = tf.train.Saver(var_list=tf.global_variables())
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.trained_weight_dir))

            if ckpt_st is not None:
                '''restore 시에는 tf.global_variables_initializer() 가 필요 없다.'''
                print(ckpt_st.model_checkpoint_path)
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            # kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'right/output_layer/logit/W_conv2d')[0]
            # print(sess.run(kernel))

            tot_pred = []
            tot_prob = []
            file_names = sess.run(file_names)

            for _ in range(0, file_names.shape[0], flags.FLAGS.batch_size):
                low_batch_x, mid_batch_x, high_batch_x = sess.run([low_batch_img, mid_batch_img, high_batch_img])
                prob = model.predict(low_res_X=low_batch_x,
                                     mid_res_X=mid_batch_x,
                                     high_res_X=high_batch_x)
                tot_prob.append(np.max(prob, axis=1).tolist())
                tot_pred.append(np.argmax(prob, axis=1).tolist())

            '''Ensemble Prediction'''
            tot_pred = np.array(tot_pred).flatten()
            for pred in tot_pred:
                print(pred, end=',')
            print('')
            tot_pred = Counter(tot_pred)
            print(tot_pred.most_common(1)[0][0])

            for prob in tot_prob:
                print(prob)

            os.makedirs(flags.FLAGS.deploy_log_dir, exist_ok=True)

            '''Graph Save'''
            # tf.train.write_graph(sess.graph_def, flags.FLAGS.deploy_log_dir, 'graph.pbtxt')
            # self._saver.save(sess, os.path.join(flags.FLAGS.deploy_log_dir, 'model_graph'))
            # print('>> Graph saved')

            '''PB File Save'''
            # builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(flags.FLAGS.deploy_log_dir, 'eye_verification_param'))
            # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
            # builder.save()

neuralnet = Neuralnet()
# neuralnet.cam_test()
neuralnet.train()
# neuralnet.integration_test()
# neuralnet.unit_test()