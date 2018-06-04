import numpy as np
import os
import re
import shutil

class DataSeparator:
    '''
        Train / Test 데이터 셋으로 분리하는 클래스
    '''
    def __init__(self):
        '''
            self.asis_data_path: original dataset root path
            self.tobe_data_path: new dataset root path
        '''
        self.asis_data_path = 'D:\\100_dataset\\eye_verification\\180423(12-17)'
        self.tobe_data_path = 'D:\\100_dataset\\eye_verification\\dataset'

        ''' re.compile('.*\\\\(25-30|30)\\\\.*'): 정규식을 통해 EAR 25 이상에 해당하는 path 만 추출 '''
        self.path_reg = re.compile('.*\\\\(25-30|30)\\\\.*')
        self.class_reg = re.compile('.*\\\\(right|left)\\\\(\d+)-(\d+)\\\\.*')

        ''' original dataset path 를 담는 변수들 '''
        self.asis_full_paths = []
        self.asis_right_paths = []
        self.asis_left_paths = []

        '''
            new dataset path 를 담는 변수들
            train: 70%, test: 30%
        '''
        self.train_left_paths, self.test_left_paths = [], []
        self.train_right_paths, self.test_right_paths = [], []

    def _path_setting(self):
        '''
            새로운 데이터 셋으로 구성하기 위해 원본 이미지 파일 경로를 추출하는 함수
            :return: None
        '''
        for path, dir, filenames in os.walk(self.asis_data_path):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                if self.path_reg.search(file_path):
                    self.asis_full_paths.append(file_path)

        self.asis_full_paths = sorted(self.asis_full_paths)
        self.asis_left_paths = np.array(self.asis_full_paths[:int(len(self.asis_full_paths) / 2)])
        self.asis_right_paths = np.array(self.asis_full_paths[int(len(self.asis_full_paths) / 2):])

        data_len = self.asis_left_paths.shape[0]
        random_sort = np.random.permutation(data_len)
        self.asis_left_paths = self.asis_left_paths[random_sort]
        self.train_left_paths, self.test_left_paths = self.asis_left_paths[:int(data_len * 0.7)], self.asis_left_paths[int(data_len * 0.7):]
        self.asis_right_paths = self.asis_right_paths[random_sort]
        self.train_right_paths, self.test_right_paths = self.asis_right_paths[:int(data_len * 0.7)], self.asis_right_paths[int(data_len * 0.7):]

    def file_move(self):
        self._path_setting()

        '''train data set'''
        print('>> Train Dataset Move Start...')
        for train_path in zip(self.train_left_paths, self.train_right_paths):
            class_num = self.class_reg.search(train_path[0]).group(2)
            group_num = self.class_reg.search(train_path[0]).group(3)
            tobe_left_path = os.path.join(self.tobe_data_path, 'train', 'left', class_num)
            tobe_right_path = os.path.join(self.tobe_data_path, 'train', 'right', class_num)

            os.makedirs(tobe_left_path, exist_ok=True)
            os.makedirs(tobe_right_path, exist_ok=True)

            shutil.move(train_path[0], os.path.join(tobe_left_path, str(group_num) + '_' + os.path.basename(train_path[0])))
            shutil.move(train_path[1], os.path.join(tobe_right_path, str(group_num) + '_' + os.path.basename(train_path[1])))
        print('>> Train Dataset Move End ...')

        '''test data set'''
        print('>> Test Dataset Move Start...')
        for test_path in zip(self.test_left_paths, self.test_right_paths):
            class_num = self.class_reg.search(test_path[0]).group(2)
            group_num = self.class_reg.search(test_path[0]).group(3)
            tobe_left_path = os.path.join(self.tobe_data_path, 'test', 'left', class_num)
            tobe_right_path = os.path.join(self.tobe_data_path, 'test', 'right', class_num)

            os.makedirs(tobe_left_path, exist_ok=True)
            os.makedirs(tobe_right_path, exist_ok=True)

            shutil.move(test_path[0], os.path.join(tobe_left_path, str(group_num) + '_' + os.path.basename(test_path[0])))
            shutil.move(test_path[1], os.path.join(tobe_right_path, str(group_num) + '_' + os.path.basename(test_path[1])))
        print('>> Test Dataset Move End ...')

# ds = DataSeparator()
# ds.file_move()

import os
import re

asis_path = 'D:\\100_dataset\\eye_verification\\eye_only_v4'
tobe_path = 'D:\\100_dataset\\eye_verification\\eye_only_v5'

reg_exp = re.compile('.*\\\\(train|test)\\\\(right|left)\\\\(\d+)\\\\(.*)')

for path, dir, filenames in os.walk(asis_path):
    for filename in filenames:
        method, diredtion, number, fname = reg_exp.search(os.path.join(path, filename)).groups()
        if number == '90':
            os.makedirs(os.path.join(tobe_path, method, diredtion, '0'), exist_ok=True)
            shutil.copy(os.path.join(path, filename), os.path.join(tobe_path, method, diredtion, '0', fname))
        else:
            os.makedirs(os.path.join(tobe_path, method, diredtion, '1'), exist_ok=True)
            shutil.copy(os.path.join(path, filename), os.path.join(tobe_path, method, diredtion, '1', fname))