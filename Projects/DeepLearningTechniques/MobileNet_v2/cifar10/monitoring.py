import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
import cx_Oracle
import datetime

class Monitoring:
    train_acc_list = deque(maxlen=1800)
    train_loss_list = deque(maxlen=1800)
    valid_acc_list = deque(maxlen=1800)
    valid_loss_list = deque(maxlen=1800)

    def __init__(self, dataset, model, seq):
        self.dataset = dataset
        self.model = model
        self.seq = seq

        self._init_database()

        f = plt.figure(figsize=(10, 8))

        self.acc_subplot = plt.subplot(211)
        self.acc_subplot.set_title('Train / Validation Accuracy')
        self.acc_subplot.set_xlim(0, 1800)
        self.acc_subplot.set_ylim(0, 1)
        self.train_acc_line = Line2D([], [], color='blue', label='train_acc')
        self.valid_acc_line = Line2D([], [], color='red', label='valid_acc')
        self.acc_subplot.add_line(self.train_acc_line)
        self.acc_subplot.add_line(self.valid_acc_line)
        self.acc_subplot.legend()

        self.loss_subplot = plt.subplot(212)
        self.loss_subplot.set_title('Train / Validation Loss')
        self.loss_subplot.set_xlim(0, 1800)
        self.loss_subplot.set_ylim(0, 10)
        self.train_loss_line = Line2D([], [], color='blue', label='train_loss')
        self.valid_loss_line = Line2D([], [], color='red', label='valid_loss')
        self.loss_subplot.add_line(self.train_loss_line)
        self.loss_subplot.add_line(self.valid_loss_line)
        self.loss_subplot.legend()

        ani = anim.FuncAnimation(f, self.update, interval=5000)
        plt.show()

    def _init_database(self):
        '''
        데이터베이스 연결을 수행하는 함수
        :return: None
        '''
        # self._conn = cx_Oracle.connect('hongbog', 'hongbog0102', cx_Oracle.makedsn('172.168.0.198', 1521, 'orcl'))
        self._conn = cx_Oracle.connect('hongbog/hongbog0102@orcl')

    def _get_cursor(self):
        '''
        데이터베이스 커서를 생성하는 함수
        :return: 데이터베이스 커서, type -> cursor
        '''
        return self._conn.cursor()

    def _close_cursor(self, cur):
        '''
        데이터베이스 커서를 닫는 함수
        :param cur: 닫을 커서
        :return: None
        '''
        cur.close()

    def _close_conn(self):
        '''
        데이터베이스 연결을 해제하는 함수
        :return: None
        '''
        self._conn.close()

    def _mon_data_from_db(self):
        cur = self._get_cursor()
        cur.execute('select train_type, acc, loss from neuralnet_train_log where dataset = :dataset and model = :model and seq = :seq order by train_type, epoch',
                    [self.dataset, self.model, self.seq])

        train_mon_log = []
        valid_mon_log = []
        for train_type, acc, loss in cur.fetchall():
            if train_type == 'train':
                train_mon_log.append([acc, loss])
            else:
                valid_mon_log.append([acc, loss])

        tobe_train_log_cnt = len(train_mon_log)
        tobe_valid_log_cnt = len(valid_mon_log)

        asis_train_log_cnt = len(self.train_acc_list)
        asis_valid_log_cnt = len(self.valid_acc_list)

        if (asis_train_log_cnt == 0) or (tobe_train_log_cnt > asis_train_log_cnt):
            for log in train_mon_log[asis_train_log_cnt:]:
                acc, loss = log
                self.train_acc_list.append(float(acc))
                self.train_loss_list.append(float(loss))

        if (asis_valid_log_cnt == 0) or (tobe_valid_log_cnt > asis_valid_log_cnt):
            for log in valid_mon_log[asis_valid_log_cnt:]:
                acc, loss = log
                self.valid_acc_list.append(float(acc))
                self.valid_loss_list.append(float(loss))

    def update(self, f):
        self._mon_data_from_db()

        self.train_acc_line.set_data([epoch for epoch in range(1, len(self.train_acc_list)+1)], self.train_acc_list)
        self.train_loss_line.set_data([epoch for epoch in range(1, len(self.train_loss_list)+1)], self.train_loss_list)
        self.valid_acc_line.set_data([epoch for epoch in range(1, len(self.valid_acc_list)+1)], self.valid_acc_list)
        self.valid_loss_line.set_data([epoch for epoch in range(1, len(self.valid_loss_list) + 1)], self.valid_loss_list)

mon = Monitoring(dataset='cifar10', model='mobilenet-v2', seq=11)