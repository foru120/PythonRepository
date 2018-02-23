import numpy as np
from PIL import Image
import os
import re
import tensorflow as tf

class Trainer:
    def __init__(self, data_loader, img_size, batch_size, n_epoch, learning_rate, drop_out_rate, decay_rate,
                 model_save_path, n_class, act_func):
        self.data_loader = data_loader
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.drop_out_rate = drop_out_rate
        self.n_epoch = n_epoch
        self.decay_rate = decay_rate
        self.model_save_path = model_save_path + '/Focal_FCN.ckpt'
        self.net = Nets(data_shape=self.img_size, channel=1, n_class=n_class, act_func=act_func, batch_size=batch_size)
        print('>> Trainer Initialized')

    def _get_optimizer(self, global_step, num_lists):

        if num_lists * 2 > 2000:
            decay_step = num_lists * 2
        else:
            decay_step = 1000

        decay_rate = self.decay_rate

        learning_rate = self.learning_rate

        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                     global_step=global_step, decay_steps=decay_step,
                                                                     decay_rate=decay_rate, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=exponential_decay_learning_rate).minimize(self.net.cost,
                                                                                                        global_step=global_step)

    def train_(self):
        num_train = self.data_loader.tr_count // self.batch_size + 1
        num_valid = self.data_loader.val_count // self.batch_size + 1

        print('>> num_train:', num_train)
        print('>> num_valid:', num_valid)
        global_step = tf.Variable(0, trainable=False)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._get_optimizer(global_step, num_train)

        train_input, train_label = self.data_loader.train_fn()
        valid_input, valid_label = self.data_loader.valid_fn()

        with tf.Session() as sess:
            print('>> Tensorflow session built. variables initialized')
            sess.run(tf.global_variables_initializer())
            # merge = tf.summary.merge_all()

            saver = tf.train.Saver()

            ckpt_st = tf.train.get_checkpoint_state(self.model_save_path)

            if ckpt_st is not None:
                saver.restore(sess, ckpt_st.model_checkpoint_path)

            tf.train.start_queue_runners(sess=sess)
            print('>> running started')
            for epoch in range(self.n_epoch):
                stt = time.time()
                total_loss, total_acc = 0., 0.

                for tstep in range(num_train):
                    st = time.time()
                    t_input, t_label = sess.run([train_input, train_label])
                    if tstep == 0:
                        print('>> t_input shape : ', t_input.shape, 't_label shape : ', t_label.shape)

                    feed_dict = {self.net.x: t_input,
                                 self.net.y: t_label,
                                 self.net.training: True,
                                 self.net.keep_prob: self.drop_out_rate}

                    loss, _ = sess.run([self.net.cost, self.optimizer], feed_dict=feed_dict)
                    et = time.time()
                    t = et - st
                    print(">> [Training] [%d/%d] step/e: %d  Loss: %.4f Step_Time: %.1f" % (
                    epoch, self.n_epoch, tstep, loss, t))
                edt = time.time()
                for vstep in range(num_valid):
                    v_input, v_label = sess.run([valid_input, valid_label])
                    feed_dict = {self.net.x: v_input,
                                 self.net.y: v_label,
                                 self.net.training: False,
                                 self.net.keep_prob: self.drop_out_rate}

                    vloss, predictions, accuracy = sess.run([
                        self.net.cost, self.net.predict, self.net.accuracy], feed_dict=feed_dict)

                    total_loss += vloss
                    total_acc += accuracy

                total_loss /= num_valid
                total_acc /= num_valid

                print(
                    '>> [total/epoch] [%d/%d] / validation avg acc: %.4f / validation avg loss: %.4f / Training time(sec): %.2f' % (
                    epoch, self.n_epoch, total_acc, total_loss, edt - stt))
                # train_writer = tf.summary.FileWriter('../log/test.summaries', sess.graph)

                saver.save(sess, self.model_save_path)
                print('>> model saved')

        sess.close()
        tf.reset_default_graph()

class TrainDataLoader:
    def __init__(self, n_validation, batch_size, img_size, data_path):
        # 밸리데이션 데이터셋 비율 ex) 5
        self.n_validation = n_validation
        self.batch_size = batch_size
        self.img_size = img_size

        # 각 환자별로 폴더가 정리되어 있는 경우 다수의 폴더가 로드되는 경우 경로 리스트
        self.data_path = [os.path.join(data_path, dir) for dir in os.listdir(data_path)]

        # 전체 세트 수 ex) 150
        self.data_path_length = len(self.data_path)

        # 밸리데이션 데이터 개수
        if round(self.data_path_length * (self.n_validation / 100)) == 0:
            self.val_data_cnt = 1
        else:
            self.val_data_cnt = round(self.data_path_length * (self.n_validation / 100))

        # 트레인 데이터셋 밸리데이션 데이터셋 분리
        self.train_data_list = self.data_path[:-self.val_data_cnt]
        self.val_data_list = self.data_path[-self.val_data_cnt:]

        # 트레인 데이터셋 이미지 경로
        self.tr_x, self.tr_y, self.tr_len = self.get_input_label(self.train_data_list)
        # print(self.tr_x)
        self.tr_x_list, self.tr_y_list, _ = self.get_list(self.train_data_list)

        # 밸리데이션 데이터셋 이미지 경로
        self.val_x, self.val_y, self.val_len = self.get_input_label(self.val_data_list)
        self.val_x_list, self.val_y_list, _ = self.get_list(self.val_data_list)

        self.tr_count = self.tr_len
        self.val_count = self.val_len
        print('Data Loader Initialized')
        print('>> Train Data List count:', self.tr_count)
        print('>> Validation Data List count:', self.val_count)

    def _try_int(self, ss):
        try:
            return int(ss)
        except:
            return ss

    def _number_key(self, s):
        return [self._try_int(ss) for ss in re.split('([0-9]+)', s)]

    # 파일명 번호 순으로 정렬
    def _sort_by_number(self, files):
        files.sort(key=self._number_key)
        return files

    # List를 Tensor로 변환
    def toTensor(self, lists):
        return tf.convert_to_tensor(lists, dtype=tf.string)

    # 데이터 List를 Tensor로 변환해서 리턴
    def get_input_label(self, lists):
        inputs, labels, lens = self.get_list(lists)
        return self.toTensor(inputs), self.toTensor(labels), lens

    # 파일경로에서 데이터경로 다 불러옴
    def get_list(self, list):
        # 데이터셋 경로를 담아 둘 빈 리스트 생성
        image_list = []
        label_list = []

        # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
        for data_path in list:
            for root, dirs, files in os.walk(data_path):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if '/x' in dir_path:
                        if len(os.listdir(dir_path)) != 0:
                            x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                            y_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                            y_path_list = [path.replace('/x/', '/y/') for path in y_path_list]

                            images_files = self._sort_by_number(x_path_list)
                            labels_files = self._sort_by_number(y_path_list)

                            for image in images_files:
                                image_list.append(image)
                                # print('xdata:', image)

                            for label in labels_files:
                                label_list.append(label)
                                # print('ydata:', label)

        return image_list, label_list, len(image_list)


    # 트레인 데이터 리더
    def tr_img_reader(self, x_path, y_path):
        # 데이터 로드
        img_str = tf.read_file(x_path)
        lab_str = tf.read_file(y_path)

        # 이미지 데이터 불러와서 동일사이즈로 리사이징
        _image = tf.image.decode_png(img_str, channels=1)
        _image = tf.image.resize_images(_image, [300, 300])
        _label = tf.image.decode_png(lab_str, channels=1)
        _label = tf.image.resize_images(_label, [300, 300])

        # 이미지 데이터를 랜덤 크롭. 같은 위치를 크롭하기 위해 컨캣
        concat_data = tf.concat([_image, _label], axis=2)
        cropped_img = tf.random_crop(value=concat_data, size=(256, 256, 2))

        # 이미지 전처리. 좌우상하 랜덤플립, 로테이션
        processed_img = tf.image.random_flip_left_right(cropped_img)
        processed_img = tf.image.random_flip_up_down(processed_img)
        sel = tf.random_uniform([], maxval=4, dtype=tf.int32)
        processed_img = tf.image.rot90(processed_img, k=sel)

        # 원래대로 데이터를 스플릿
        split_image, split_label = tf.split(axis=2, num_or_size_splits=2, value=processed_img)

        # 백그라운드가 1, 포어그라운드가 0인 라벨데이터 생성
        labels_back = tf.subtract(tf.ones_like(split_label) * 255, split_label)
        split_label = tf.to_float(tf.concat([split_label, labels_back], axis=2))

        # min-max scaling
        scaled_img = tf.divide(tf.subtract(split_image, tf.reduce_min(split_image)),
                               tf.subtract(tf.reduce_max(split_image), tf.reduce_min(split_image)))
        scaled_label = tf.divide(tf.subtract(split_label, tf.reduce_min(split_label)),
                                 tf.subtract(tf.reduce_max(split_label), tf.reduce_min(split_label)))

        return scaled_img, scaled_label

    # 밸리데이션 데이터 리더
    def val_img_reader(self, x_path, y_path):
        # 데이터 로드
        img_str = tf.read_file(x_path)
        lab_str = tf.read_file(y_path)

        # 이미지 데이터 불러와서 동일사이즈로 리사이징
        _image = tf.image.decode_png(img_str, channels=1)
        _image = tf.image.resize_images(_image, [300, 300])
        _label = tf.image.decode_png(lab_str, channels=1)
        _label = tf.image.resize_images(_label, [300, 300])

        # 이미지 데이터를 랜덤 크롭. 같은 위치를 크롭하기 위해 컨캣
        concat_data = tf.concat([_image, _label], axis=2)
        cropped_img = tf.random_crop(value=concat_data, size=(256, 256, 2))

        # 원래대로 데이터를 스플릿
        split_image, split_label = tf.split(axis=2, num_or_size_splits=2, value=cropped_img)

        # 백그라운드가 1, 포어그라운드가 0인 라벨데이터 생성
        labels_back = tf.subtract(tf.ones_like(split_label) * 255, split_label)
        split_label = tf.to_float(tf.concat([split_label, labels_back], axis=2))

        # min-max scaling
        scaled_img = tf.divide(tf.subtract(split_image, tf.reduce_min(split_image)), tf.subtract(tf.reduce_max(split_image), tf.reduce_min(split_image)))
        scaled_label = tf.divide(tf.subtract(split_label, tf.reduce_min(split_label)), tf.subtract(tf.reduce_max(split_label), tf.reduce_min(split_label)))

        return scaled_img, scaled_label

    # 트레인 데이터 로더
    def train_loader(self):
        # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
        dataset = tf.contrib.data.Dataset.from_tensor_slices((self.tr_x, self.tr_y)).repeat()

        # 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다
        dataset_map = dataset.map(self.tr_img_reader).batch(self.batch_size)

        # 데이터셋을 이터레이터를 통해 지속적으로 불러준다
        iterator = dataset_map.make_one_shot_iterator()

        # 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다.
        batch_input, batch_label = iterator.get_next()

        return batch_input, batch_label

    # 밸리데이션 데이터 로더
    def val_loader(self):
        # 데이터셋을 불러오는데 불러오는 데이터는 텐서타입이어서 배치단위로 계속 부르기 위해 텐서슬라이스 함수를 반복적으로 사용한다.
        dataset = tf.contrib.data.Dataset.from_tensor_slices((self.val_x, self.val_y)).repeat()

        # 데이터셋의 맵함수를 통해 배치사이즈별로 잘라내는데 사용하는 함수를 맵함수 안에 넣어준다
        dataset_map = dataset.map(self.val_img_reader).batch(self.batch_size)

        # 데이터셋을 이터레이터를 통해 지속적으로 불러준다
        iterator = dataset_map.make_one_shot_iterator()

        # 세션이 런 될 때마다 반복해서 이터레이터를 소환한다. 그렇게 해서 다음 배치 데이터셋을 불러온다.
        batch_input, batch_label = iterator.get_next()

        return batch_input, batch_label

    # 트레인 배치 로더
    def train_fn(self):
        with tf.device('/cpu:0'):
            image_batch, label_batch = self.train_loader()
            return image_batch, label_batch

    # 밸리데이션 배치 로더
    def valid_fn(self):
        with tf.device('/cpu:0'):
            image_batch, label_batch = self.val_loader()
            return image_batch, label_batch

BLUR_IMAGE_PATH = 'D:\\Data\\casia_blurring\\image_data'

loader = TrainDataLoader(n_validation=10, batch_size=100, img_size=256, data_path=BLUR_IMAGE_PATH)
trainer = Trainer(data_loader=loader, img_size=256, n_class=2, batch_size=100, n_epoch=20, learning_rate=0.001, drop_out_rate=0.01, decay_rate=0.96, model_save_path='/home/bjh/new/2ds/model1/models', act_func='ce')
trainer.train_()