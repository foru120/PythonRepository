import cx_Oracle
import datetime

class Database:

    def __init__(self):
        self._init_conn()

    def _init_conn(self):
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

    def mon_data_to_db(self, model, train_type, seq, epoch, acc, loss, train_time):
        '''
        모니터링 항목 DB로 저장
        :param model: 신경망 모델 명
        :param train_type: 훈련 타입 (train / validation / test)
        :param seq: 훈련 번호
        :param epoch: 훈련 에폭
        :param acc: 훈련 정확도
        :param loss: 훈련 손실 값
        :param train_time: 훈련 시간
        :param log_time: 로그 기록 시간
        :return:
        '''
        cur = self._get_cursor()
        cur.execute('insert into imagenet_train_log values(:model, :train_type, :seq, :epoch, :acc, :loss, :train_time, :log_time)',
                    [model, train_type, seq, epoch, acc, loss, train_time, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        self._conn.commit()
        self._close_cursor(cur)

    def mon_data_to_file(self, train_psnr_mon, train_loss_mon, valid_psnr_mon, valid_loss_mon):
        '''
        모니터링 대상(훈련 PSNR, 훈련 손실 값, 검증 PSNR, 검증 손실 값) 파일로 저장
        :param train_acc_mon: 훈련 PSNR
        :param train_loss_mon: 훈련 손실 값
        :param valid_acc_mon: 검증 PSNR
        :param valid_loss_mon: 검증 손실 값
        :return: None
        '''
        with open('D:/Source/PythonRepository/Hongbog/SuperResolution//mon_log/mon_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.txt', 'a') as f:
            f.write(','.join([str(train_psnr_mon), str(valid_psnr_mon), str(train_loss_mon), str(valid_loss_mon)]) + '\n')