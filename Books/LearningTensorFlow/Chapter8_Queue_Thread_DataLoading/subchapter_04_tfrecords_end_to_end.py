import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np

class DataLoader:
    def __init__(self):
        self.save_dir = './Mnist_data/tfrecord'
        self.data_splits = ['train', 'test', 'validation']

    def create_mnist(self):
        # save_dir 에 데이터를 내려받는다.
        self.data_sets = mnist.read_data_sets(self.save_dir,
                                              dtype=tf.uint8,
                                              reshape=False,
                                              validation_size=1000)

        for d in range(len(self.data_splits)):
            print('saving: ' + self.data_splits[d])
            data_set = self.data_sets[d]

            filename = os.path.join(self.save_dir, self.data_splits[d] + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(filename)

            for index in range(data_set.images.shape[0]):
                image = data_set.images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[1]])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[2]])),
                    'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[3]])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.labels[index]])),
                    'image_raw': tf.train.Feature(int64_list=tf.train.BytesList(value=[image]))
                }))
                writer.write(example.SerializeToString())

            writer.close()

    def read_mnist(self):
        filename = os.path.join(self.save_dir, self.data_splits[0] + '.tfrecords')
        filename_queue = tf.train.string_input_producer([filename], num_epochs=10)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized=serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }
        )

        image = tf.decode_raw(features['image_raw'], tf.uint8)  # 원본 바이트 스트링 데이터를 디코딩
        image.set_shape([784])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # 픽셀 값 Normalization
        label = tf.cast(features['label'], tf.int32)

        # 내부적으로 RandomShuffleQueue 를 사용해 이 큐에 batch_size + min_after_dequeue 만큼의 항목이 쌓일 때까지 인스턴스를 쌓는다.
        images_batch, labels_batch = tf.train.shuffle_batch(  # 랜덤 셔플 후 배치 처리
            tensors=[image, label],
            batch_size=128,
            capacity=2000,
            min_after_dequeue=1000
        )

        return images_batch, labels_batch

    def model(self, image_batch, labels_batch):
        W = tf.get_variable('W', [28*28, 10])
        y_pred = tf.matmul(image_batch, W)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                   labels=labels_batch)
        self.loss = tf.reduce_mean(self.loss)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self):
        image_batch, labels_batch = self.read_mnist()

        self.model(image_batch, labels_batch)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                step = 0
                while not coord.should_stop():
                    step += 1
                    sess.run(self.train_op)
                    if step % 500 == 0:
                        loss_mean_val = sess.run(self.loss)
                        print('>> [%10d], loss: %.4f' % (step, loss_mean_val))
            except tf.errors.OutOfRangeError:  # 큐가 비어서 발생하는 에러
                print('Done training for %d steps.' % (step))
            finally:
                coord.request_stop()  # 쓰레드 정지 요청

            coord.join(threads)

dataloader = DataLoader()
dataloader.train()