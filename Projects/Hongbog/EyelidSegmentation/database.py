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

    def close_conn(self):
        '''
        데이터베이스 연결을 해제하는 함수
        :return: None
        '''
        self._conn.close()

    def mon_data_to_file(self, train_acc_mon, train_loss_mon, valid_acc_mon, valid_loss_mon):
        '''
        모니터링 대상(훈련 정확도, 훈련 손실 값, 검증 정확도, 검증 손실 값) 파일로 저장
        :param train_acc_mon: 훈련 정확도
        :param train_loss_mon: 훈련 손실 값
        :param valid_acc_mon: 검증 정확도
        :param valid_loss_mon: 검증 손실 값
        :return: None
        '''
        with open(self._FLAGS.mon_data_log_path, 'a') as f:
            f.write(','.join([str(train_acc_mon), str(valid_acc_mon), str(train_loss_mon), str(valid_loss_mon)]) + '\n')

    def mon_data_to_db(self, train_acc_mon, train_loss_mon, valid_acc_mon, valid_loss_mon, train_time):
        '''
        모니터링 대상(훈련 정확도, 훈련 손실 값, 검증 정확도, 검증 손실 값) DB로 저장
        :param train_acc_mon: 훈련 정확도
        :param train_loss_mon: 훈련 손실 값
        :param valid_acc_mon: 검증 정확도
        :param valid_loss_mon: 검증 손실 값
        :return: None
        '''
        cur = self._get_cursor()
        cur.execute('insert into eyelid_seg_log values(:train_log, :log_time, :train_time, :train_acc, :valid_acc, :train_loss, :valid_loss)',
                    [self.train_log, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_time, train_acc_mon, valid_acc_mon, train_loss_mon, valid_loss_mon])
        self._conn.commit()
        self._close_cursor(cur)