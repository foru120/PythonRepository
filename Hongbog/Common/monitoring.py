import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
import cx_Oracle
import datetime

class Monitoring:
    train_acc_list = deque(maxlen=100)
    train_loss_list = deque(maxlen=100)
    valid_acc_list = deque(maxlen=100)
    valid_loss_list = deque(maxlen=100)

    def __init__(self, load_type):
        self.load_type = load_type

        if load_type == 'db':
            self._init_database()
            self._get_max_log_num()

        f = plt.figure(figsize=(10, 8))

        self.acc_subplot = plt.subplot(211)
        self.acc_subplot.set_title('Accuracy')
        self.acc_subplot.set_xlim(0, 100)
        self.acc_subplot.set_ylim(0, 100)
        self.a1 = Line2D([], [], color='black', label='train')
        self.a2 = Line2D([], [], color='blue', label='validation')
        self.acc_subplot.add_line(self.a1)
        self.acc_subplot.add_line(self.a2)
        self.acc_subplot.legend()

        self.loss_subplot = plt.subplot(212)
        self.loss_subplot.set_title('Loss')
        self.loss_subplot.set_xlim(0, 100)
        self.loss_subplot.set_ylim(0, 10)
        self.l1 = Line2D([], [], color='black', label='train')
        self.l2 = Line2D([], [], color='blue', label='validation')
        self.loss_subplot.add_line(self.l1)
        self.loss_subplot.add_line(self.l2)
        self.loss_subplot.legend()

        ani = anim.FuncAnimation(f, self.update, interval=5000)
        plt.show()

    def _init_database(self):
        '''
        데이터베이스 연결을 수행하는 함수
        :return: None
        '''
        self._conn = cx_Oracle.connect('hongbog/hongbog0102@localhost:1521/orcl')

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

    def _get_max_log_num(self):
        '''
        현재 로깅된 최대 로그 숫자를 DB 에서 가져오는 함수
        :return: None
        '''
        cur = self._get_cursor()
        cur.execute('select nvl(max(log_num), 0) max_log_num from train_log')
        self._max_log_num = cur.fetchone()[0]
        self._close_cursor(cur)

    def _mon_data_from_db(self):
        cur = self._get_cursor()
        cur.execute('select train_acc, valid_acc, train_loss, valid_loss from train_log where log_num = :log_num order by log_time', [self._max_log_num+1])

        mon_log = []
        for train_acc, valid_acc, train_loss, valid_loss in cur.fetchall():
            mon_log.append([train_acc, valid_acc, train_loss, valid_loss])

        mon_log_cnt = len(mon_log)
        cur_mon_cnt = len(self.train_acc_list)

        if (cur_mon_cnt == 0) or (mon_log_cnt > cur_mon_cnt):
            for log in mon_log[cur_mon_cnt:]:
                train_acc_mon, valid_acc_mon, train_loss_mon, valid_loss_mon = log
                self.train_acc_list.append(float(train_acc_mon) * 100)
                self.train_loss_list.append(float(train_loss_mon))
                self.valid_acc_list.append(float(valid_acc_mon) * 100)
                self.valid_loss_list.append(float(valid_loss_mon))

    def _mon_data_from_file(self):
        try:
            mon_log = []

            with open('D:/Source/PythonRepository/Hongbog/Preprocessing/mon_log/mon_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') +'.txt', 'r') as f:
                mon_log = [log for log in f]

            mon_log_cnt = len(mon_log)
            cur_mon_cnt = len(self.train_acc_list)

            if (cur_mon_cnt == 0) or (mon_log_cnt > cur_mon_cnt):
                for log_text in mon_log[cur_mon_cnt:]:
                    train_acc_mon, valid_acc_mon, train_loss_mon, valid_loss_mon = log_text.rstrip().split(',')
                    self.train_acc_list.append(float(train_acc_mon)*100)
                    self.train_loss_list.append(float(train_loss_mon))
                    self.valid_acc_list.append(float(valid_acc_mon)*100)
                    self.valid_loss_list.append(float(valid_loss_mon))
        except FileNotFoundError as e:
            print(e)

    def update(self, f):
        if self.load_type == 'file':
            self._mon_data_from_file()
        else:
            self._mon_data_from_db()
        self.a1.set_data([epoch for epoch in range(1, len(self.train_acc_list)+1)], self.train_acc_list)
        self.a2.set_data([epoch for epoch in range(1, len(self.valid_acc_list)+1)], self.valid_acc_list)
        self.l1.set_data([epoch for epoch in range(1, len(self.train_loss_list)+1)], self.train_loss_list)
        self.l2.set_data([epoch for epoch in range(1, len(self.valid_loss_list)+1)], self.valid_loss_list)

mon = Monitoring(load_type='db')