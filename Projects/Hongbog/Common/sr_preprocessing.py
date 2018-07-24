import os
import re
from PIL import Image
import cv2
import numpy as np
from collections import defaultdict
import shutil

#todo Data Generation Preprocessing
'''
    Super Resolution 의 해상도를 더 높이기 위해 원본 이미지와 추출된 고 해상도 이미지를 데이터 셋으로 만드는 기능.
'''

class DataGeneration:

    def image_blurring(self, asis_path, tobe_path):
        '''
        원본 이미지를 4 가지 blur 방식을 사용해 이미지 생성 (Averaging, Gaussian, Median, Bilateral)
        :param asis_path: 원본 이미지 경로
        :param tobe_path: blur 이미지가 저장될 경로
        :return: None
        '''
        for img_path in self.get_file_path_list(asis_path, []):
            folder_name = re.match('\\\(.*)\\\(.*\.bmp)', img_path[img_path.index(asis_path) + len(asis_path):])
            if folder_name is not None:
                img = cv2.imread(img_path)
                height, width = img.shape[:2]
                if width == 640 and height == 480:
                    os.makedirs(os.path.join(tobe_path, folder_name[1]), exist_ok=True)
                    file_name, ext = os.path.splitext(folder_name[2])

                    blur_img = cv2.cvtColor(cv2.blur(img, (7, 7)), cv2.COLOR_BGR2GRAY)
                    gaussian_img = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 1), cv2.COLOR_BGR2GRAY)
                    median_img = cv2.cvtColor(cv2.medianBlur(img, 7), cv2.COLOR_BGR2GRAY)
                    bilateral_img = cv2.cvtColor(cv2.bilateralFilter(img, 9, 75, 75), cv2.COLOR_BGR2GRAY)

                    cv2.imwrite(os.path.join(tobe_path, folder_name[1], file_name + '_blur' + ext), blur_img)
                    cv2.imwrite(os.path.join(tobe_path, folder_name[1], file_name + '_gaussian' + ext), gaussian_img)
                    cv2.imwrite(os.path.join(tobe_path, folder_name[1], file_name + '_median' + ext), median_img)
                    cv2.imwrite(os.path.join(tobe_path, folder_name[1], file_name + '_bilateral' + ext), bilateral_img)

    def get_file_path_list(self, root_path, path_list):
        '''
        특정 폴더 밑에 있는 파일 경로를 리스트화하는 함수 (os.walk() 참조)
        :param root_path: 파일이 포함된 상위 경로
        :param path_list: 파일의 경로 정보를 담을 리스트
        :param mode: 'F' -> (Full) 전체 이미지 폴더를 대상으로 파일 검출,  'E' -> (Edge), edge 와 non-edge 폴더를 대상으로 파일 검출
        :return: 이미지 경로, type -> list
        '''
        for leaf_path in os.listdir(root_path):
            full_path = os.path.join(root_path, leaf_path)
            if os.path.isdir(full_path):
                path_list = self.get_file_path_list(full_path, path_list)
            elif os.path.isfile(full_path):
                path_list.append(full_path)
        return path_list

    def image_gray_scale_extraction(self, img_path):
        '''
        경로에 해당하는 이미지 파일의 GrayScale 값을 추출하는 함수
        :param img_path: 이미지 경로
        :return: 이미지에서 추출된 GrayScale 값, type -> list
        '''
        img_data = []
        x_pixel, y_pixel = (64, 48)
        gray_img = Image.open(img_path).convert('L')  # RGB -> Image.open(img_path).convert('RGB')
        for y in range(0, y_pixel):
            for x in range(0, x_pixel):
                img_data.append(gray_img.getpixel((x, y)))
        return img_data

    def data_to_file(self, data_path, data, name):
        '''
        추출된 이미지 값을 file 로 저장하는 함수
        :param img_data: 추출된 이미지 값
        :param name: 파일 이름
        :return: None
        '''
        with open(os.path.join(data_path, str(name)+'.txt'), mode='w') as file:
            for idx in np.random.permutation(len(data)):
                file.write(','.join([str(d) for d in data[idx][0]]) + ',' + ','.join([str(d) for d in data[idx][1]]) + '\n')

    def image_to_data(self, asis_path, tobe_path):
        '''
        만들어진 패치 크기 이미지의 Gray Scale 값을 파일로 저장하는 함수
        :param asis_path: 저 해상도 이미지 경로
        :param tobe_path: 고 해상도 이미지 경로
        :return: None
        '''
        tobe_img_data, cnt_per_batch, file_cnt = defaultdict(dict), 100, 1
        blur_sort = ['bilateral', 'blur', 'gaussian', 'median']

        for idx, tobe_img_path in enumerate(self.get_file_path_list(os.path.join(tobe_path, 'cropped'), [])):
            tobe_img_gray_values = self.image_gray_scale_extraction(tobe_img_path)
            m = re.match('(.*)\\\(.*)\\\(.*)\\\(.*)\\\(.*)\.bmp', tobe_img_path).groups()
            path1, path2, path3, ori_filename = m[1:]
            tobe_img_data[path1 + ',' + path2 + ',' + path3][ori_filename] = tobe_img_gray_values

            if (idx+1)%cnt_per_batch == 0:
                tot_image_data = []
                delete_keys = []
                for key in tobe_img_data.keys():
                    if len(tobe_img_data[key].keys()) == 100:
                        delete_keys.append(key)

                for key in delete_keys:
                    blur_img_paths = []
                    for sort in blur_sort:
                        p1, p2, p3 = key.split(',')
                        blur_img_paths.append(os.path.join(asis_path, 'cropped', p1, p2, p3 + '_' + sort))

                    for blur_img_path in blur_img_paths:
                        for path in self.get_file_path_list(blur_img_path, []):
                            blur_filename = re.match('.*\\\(.*)\.bmp', path).groups()[0]
                            blur_img_gray_values = self.image_gray_scale_extraction(path)
                            tot_image_data.append([blur_img_gray_values, tobe_img_data[key][blur_filename]])

                    tobe_img_data.pop(key)

                for idx in range(0, len(tot_image_data), 100):
                    print('create file, ' + str(file_cnt) + '.txt')
                    self.data_to_file('D:\\Data\\casia_super_resolution\\training_data\\level_3\\image_data', tot_image_data[idx:idx+100], file_cnt)
                    file_cnt = file_cnt + 1

    def create_patch_image(self, root_folder, branch_folder, file_name, asis_path, tobe_path):
        '''
        패치 사이즈 단위로 이미지 잘라내는 함수
        :param root_folder: 새로 생성된 패치 이미지 디렉토리의 최 상위 경로
        :param branch_folder: 새로 생성된 패치 이미지 디렉토리의 중간 분기 경로
        :param file_name: 새로 생성된 패치 이미지 디렉토리의 상위 경로
        :param ori_path: 원본 이미지 경로
        :return: None
        '''
        x_pixel, y_pixel = 640, 480  # x_pixel, y_pixel: 입력 이미지 x축, y축 픽셀 값
        x_delta, y_delta, img_cnt = 64, 48, 1  # x_delta, y_delta: 이미지 x축, y축 분할 단위, img_cnt: 이미지 순번을 위한 변수

        ori_img = Image.open(asis_path)  # rgb_im = im.convert('RGB')
        width, height = ori_img.size

        if (width == x_pixel) and (height == y_pixel):
            for init_y in range(0, y_pixel, y_delta):
                for init_x in range(0, x_pixel, x_delta):
                    new_img = Image.new('L', (x_delta, y_delta))  # GrayScale 이미지로 생성, Image.new('RGB', (width, height)): RGB 이미지로 생성
                    for y in range(init_y, init_y + y_delta):
                        for x in range(init_x, init_x + x_delta):
                            new_img.putpixel((x % x_delta, y % y_delta), ori_img.getpixel((x, y)))
                    new_img.save(os.path.join(tobe_path, root_folder, branch_folder, file_name, str(img_cnt)+'.bmp'))
                    img_cnt = img_cnt + 1

    def start_image_patching(self):
        '''
        이미지 패치 관련 Main 함수
        :return: None
        '''
        asis_path = 'D:\\Data\\casia_super_resolution\\2th_super_resolution\\non-cropped'
        tobe_path = 'D:\\Data\\casia_super_resolution\\2th_super_resolution\\cropped'

        path_list = self.get_file_path_list(asis_path, [])

        for path in path_list:
            root_folder, branch_folder, file_name = os.path.splitext(path)[0].split(os.path.sep)[-3: ]
            os.makedirs(os.path.join(tobe_path, root_folder, branch_folder, file_name), exist_ok=True)
            self.create_patch_image(root_folder, branch_folder, file_name, path, tobe_path)

    def sparse_file_move(self, asis_path, tobe_path):
        '''
        특정 파일만 남기고 나머지 파일을 이동시키는 함수
        :param asis_path: 이미지 원본 경로
        :param tobe_path: 이미지 이동 경로
        :return: None
        '''
        file_num = sorted([num + n for n in range(11, 100) for num in range(0, 9597, 100)])

        for num in file_num:
            try:
                shutil.move(os.path.join(asis_path, str(num) + '.txt'), os.path.join(tobe_path, str(num) + '.txt'))
            except FileNotFoundError:
                print(str(num) + '.txt, 파일이 존재하지 않습니다.')

dataGen = DataGeneration()
# dataGen.start_image_patching()
# dataGen.image_to_data('D:\\Data\\casia_super_resolution\\casia_blurring',
#                       'D:\\Data\\casia_super_resolution\\2th_super_resolution')
dataGen.sparse_file_move('D:\\Data\\casia_super_resolution\\training_data\\level_3\\image_data',
                         'D:\\Data\\casia_super_resolution\\training_data\\level_3\\backup')