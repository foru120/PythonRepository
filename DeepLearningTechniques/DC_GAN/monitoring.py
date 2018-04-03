import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
import cx_Oracle
import datetime

class Monitoring:
    train_gloss_list = deque(maxlen=1000)
    train_dloss_list = deque(maxlen=1000)

    def __init__(self, load_type, train_log):
        self.load_type = load_type
        self.train_log = train_log

        if load_type == 'db':
            self._init_database()

        f = plt.figure(figsize=(10, 8))

        self.acc_subplot = plt.subplot(211)
        self.acc_subplot.set_title('Generator Loss')
        self.acc_subplot.set_xlim(0, 1000)
        self.acc_subplot.set_ylim(0, 3000)
        self.g = Line2D([], [], color='black', label='g_loss')
        self.acc_subplot.add_line(self.g)
        self.acc_subplot.legend()

        self.loss_subplot = plt.subplot(212)
        self.loss_subplot.set_title('Discriminator Loss')
        self.loss_subplot.set_xlim(0, 1000)
        self.loss_subplot.set_ylim(0, 2000)
        self.d = Line2D([], [], color='blue', label='d_loss')
        self.loss_subplot.add_line(self.d)
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

    def _mon_data_from_db(self):
        cur = self._get_cursor()
        cur.execute('select train_gloss, train_dloss from dcgan_log where train_log = :log_num order by log_time', [self.train_log])

        mon_log = []
        for train_gloss, train_dloss in cur.fetchall():
            mon_log.append([train_gloss, train_dloss])

        mon_log_cnt = len(mon_log)
        cur_mon_cnt = len(self.train_gloss_list)

        if (cur_mon_cnt == 0) or (mon_log_cnt > cur_mon_cnt):
            for log in mon_log[cur_mon_cnt:]:
                train_gloss, train_dloss = log
                self.train_gloss_list.append(float(train_gloss))
                self.train_dloss_list.append(float(train_dloss))

    def _mon_data_from_file(self):
        try:
            mon_log = []

            with open('D:/Source/PythonRepository/Hongbog/DC_GAN/mon_log/mon_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') +'.txt', 'r') as f:
                mon_log = [log for log in f]

            mon_log_cnt = len(mon_log)
            cur_mon_cnt = len(self.train_gloss_list)

            if (cur_mon_cnt == 0) or (mon_log_cnt > cur_mon_cnt):
                for log_text in mon_log[cur_mon_cnt:]:
                    train_gloss, train_dloss = log_text.rstrip().split(',')
                    self.train_gloss_list.append(float(train_gloss))
                    self.train_dloss_list.append(float(train_dloss))
        except FileNotFoundError as e:
            print(e)

    def update(self, f):
        if self.load_type == 'file':
            self._mon_data_from_file()
        else:
            self._mon_data_from_db()

        self.g.set_data([epoch for epoch in range(1, len(self.train_gloss_list)+1)], self.train_gloss_list)
        self.d.set_data([epoch for epoch in range(1, len(self.train_dloss_list)+1)], self.train_dloss_list)

mon = Monitoring(load_type='db', train_log=1)