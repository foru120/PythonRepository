# import os
# import numpy as np
# import tensorflow as tf
# import time
#
# class Dataloader:
#     def __init__(self, data_path, epoch, batch_size):
#         self.data_path = data_path
#         self.epoch = epoch
#         self.batch_size = batch_size
#
#         stime = time.time()
#         self._data_setting()
#         etime = time.time()
#         print('Data Path Loading Complete >>>', round(etime - stime, 2))
#
#     def _data_setting(self):
#         file_paths = []
#
#         for dirpath, dirnames, filenames in os.walk(self.data_path):
#             if filenames:
#                 for filename in filenames:
#                     file_paths.append(os.path.join(dirpath, filename))
#
#         self.tot_cnt = len(file_paths)
#         self.train_paths = tf.convert_to_tensor(file_paths[0: int(self.tot_cnt * 0.6)], dtype=tf.string)
#         self.train_cnt = int(self.tot_cnt * 0.6)
#         self.test_paths = file_paths[int(self.tot_cnt * 0.6): int(self.tot_cnt * 0.9)]
#         self.test_cnt = len(self.test_paths)
#         self.valid_paths = file_paths[int(self.tot_cnt * 0.9): ]
#         self.valid_cnt = len(self.valid_paths)
#
#     def _train_loader(self):
#         filename_queue = tf.train.string_input_producer(self.train_paths, shuffle=True, capacity=10)
#         reader = tf.TextLineReader()
#         self.train_key, value = reader.read(filename_queue)
#         record_defaults = [[0.] for _ in range(64*48*2)]
#         xy = tf.decode_csv(value, record_defaults=record_defaults)
#         self.train_x, self.train_y = tf.train.batch([xy[0: 64*48], xy[64*48: ]], batch_size=self.batch_size)
#
#     def _test_loader(self):
#         filename_queue = tf.train.string_input_producer(self.test_paths, shuffle=True, capacity=10)
#         reader = tf.TextLineReader()
#         self.test_key, value = reader.read(filename_queue)
#         record_defaults = [[0.] for _ in range(64*48*2)]
#         xy = tf.decode_csv(value, record_defaults=record_defaults)
#         self.test_x, self.test_y = tf.train.batch([xy[0: 64*48], xy[64*48: ]], batch_size=self.batch_size)
#
#     def _valid_loader(self):
#         filename_queue = tf.train.string_input_producer(self.valid_paths, shuffle=True, capacity=10)
#         reader = tf.TextLineReader()
#         self.valid_key, value = reader.read(filename_queue)
#         record_defaults = [[0.] for _ in range(64*48*2)]
#         xy = tf.decode_csv(value, record_defaults=record_defaults)
#         self.valid_x, self.valid_y = tf.train.batch([xy[0: 64*48], xy[64*48: ]], batch_size=self.batch_size)
#
#     def start(self):
#         self._train_loader()
#         self._valid_loader()
#         self._test_loader()
#
# config = tf.ConfigProto(
#         gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
# )
#
# with tf.Session(config=config) as sess:
#     sess.run(tf.global_variables_initializer())
#
#     loader = Dataloader(data_path='D:\\Data\\casia_blurring\\test', epoch=1, batch_size=10)
#     loader.start()
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     try:
#         '''Train Part'''
#         print('Train')
#         for epoch in range(loader.epoch):
#             for _ in range(loader.train_cnt * int(loader.data_cnt_per_file / loader.batch_size)):
#                 train_key, train_batch_x, train_batch_y = sess.run([loader.train_key, loader.train_x, loader.train_y])
#                 print('[Training]', train_key, np.array(train_batch_x).shape, np.array(train_batch_y).shape)
#
#         '''Validation Part'''
#         print('Validation')
#         for epoch in range(loader.epoch):
#             for _ in range(loader.valid_cnt * int(loader.data_cnt_per_file / loader.batch_size)):
#                 valid_key, valid_batch_x, valid_batch_y = sess.run([loader.valid_key, loader.valid_x, loader.valid_y])
#                 print('[Validation]',valid_key, np.array(valid_batch_x).shape, np.array(valid_batch_y).shape)
#
#         '''Test Part'''
#         print('Test')
#         for epoch in range(loader.epoch):
#             for _ in range(loader.test_cnt * int(loader.data_cnt_per_file / loader.batch_size)):
#                 test_key, test_batch_x, test_batch_y = sess.run([loader.test_key, loader.test_x, loader.test_y])
#                 print('[Testing]',test_key, np.array(test_batch_x).shape, np.array(test_batch_y).shape)
#     except tf.errors.OutOfRangeError as e:
#         coord.request_stop(e)
#
#     coord.request_stop()
#     coord.join(threads)