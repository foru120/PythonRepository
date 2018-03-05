import time
import tensorflow as tf
import os

class Dataloader:

    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size

        stime = time.time()
        self._data_setting()
        etime = time.time()
        print('Data Path Loading Complete >>>', round(etime - stime, 2))

    def _data_setting(self):
        file_paths = []

        for dirpath, dirnames, filenames in os.walk(self.data_path):
            if filenames:
                for filename in filenames:
                    file_paths.append(os.path.join(dirpath, filename))

        self.tot_cnt = len(file_paths)
        self.train_paths = file_paths[0: int(self.tot_cnt * 0.6)]
        self.train_cnt = len(self.train_paths)
        self.test_paths = file_paths[int(self.tot_cnt * 0.6): int(self.tot_cnt * 0.9)]
        self.test_cnt = len(self.test_paths)
        self.valid_paths = file_paths[int(self.tot_cnt * 0.9): ]
        self.valid_cnt = len(self.valid_paths)

    def _train_loader(self):
        filename_queue = tf.train.string_input_producer(self.train_paths, shuffle=True)
        reader = tf.TextLineReader()
        _, value = reader.read(filename_queue)
        record_defaults = [[0.] for _ in range(64*48*2)]
        xy = tf.decode_csv(value, record_defaults=record_defaults)
        self.train_x, self.train_y = tf.train.batch([xy[0: 64*48], xy[64*48: ]], batch_size=self.batch_size)

    def _test_loader(self):
        filename_queue = tf.train.string_input_producer(self.test_paths, shuffle=True)
        reader = tf.TextLineReader()
        _, value = reader.read(filename_queue)
        record_defaults = [[0.] for _ in range(64*48*2)]
        xy = tf.decode_csv(value, record_defaults=record_defaults)
        self.test_x, self.test_y = tf.train.batch([xy[0: 64*48], xy[64*48: ]], batch_size=self.batch_size)

    def _valid_loader(self):
        filename_queue = tf.train.string_input_producer(self.valid_paths, shuffle=True)
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        record_defaults = [[0.] for _ in range(64*48*2)]
        xy = tf.decode_csv(value, record_defaults=record_defaults)
        self.valid_x, self.valid_y = tf.train.batch([xy[0: 64*48], xy[64*48: ]], batch_size=self.batch_size)

    def start(self):
        self._train_loader()
        self._valid_loader()
        self._test_loader()