import cx_Oracle
import datetime

class Database:

    def __init__(self, FLAGS, train_log):
        self._FLAGS = FLAGS
        self.train_log = train_log

    def init_database(self):
        '''
        데이터베이스 연결을 수행하는 함수
        :return: None
        '''
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

    def close_conn(self):
        '''
        데이터베이스 연결을 해제하는 함수
        :return: None
        '''
        self._conn.close()

    def mon_data_to_db(self, epoch, train_acc, test_acc, train_loss, test_loss, train_time, test_time):
        '''
        모니터링 대상(훈련 PSNR, 훈련 손실 값, 검증 PSNR, 검증 손실 값) DB로 저장
        :param train_gloss_mon: Train Generator Loss
        :param train_dloss_mon: Train Discriminator Loss
        :param train_time: Train Time
        :return: None
        '''
        cur = self._get_cursor()
        cur.execute('insert into eye_verification_log values(:train_log, :epoch, :log_time, :train_time, :test_time, :train_acc, :test_acc, :train_loss, :test_loss)',
                    [self.train_log, epoch, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_time, test_time, train_acc, test_acc, train_loss, test_loss])
        self._conn.commit()
        self._close_cursor(cur)