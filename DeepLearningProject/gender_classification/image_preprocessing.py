import os
import numpy as np
from PIL import Image
import matplotlib.image as mimage
import math

class ImagePreprocessing:
    ASIS_IMAGE_PATH = 'D:\\03.GitHub\\PythonRepository\\DeepLearningProject\\gender_classification\\asis_image\\'
    TOBE_IMAGE_PATH = 'D:\\03.GitHub\\PythonRepository\\DeepLearningProject\\gender_classification\\tobe_image\\'

    def __init__(self):
        self.__gray_data = []  # 이미지가 Gray Scale 로 변환된 데이터
        self.__labels = dict()  # 이미지 이름 별 label 데이터
        self.__image_path = []  # 이미지 경로 데이터
        self.__female_list = []  # 여자 label list
        self.__male_list = []  # 남자 label list
        self.__image_cnt = 0  # image counting

    def _load_images(self):
        dir_list = os.listdir(ImagePreprocessing.ASIS_IMAGE_PATH + 'lfw-deepfunneled\\')  # Image Directory 목록 추출
        for dir in dir_list:  # Directory 별 이미지 리스트 추출
            for image_path in [dir + '\\' + image for image in os.listdir(ImagePreprocessing.ASIS_IMAGE_PATH + 'lfw-deepfunneled\\' + dir)]:
                self.__image_path.append(image_path)

    def _label_setting(self):
        '''
            남, 여 비율을 맞춰서 label data setting
        '''
        self.__female_list = np.loadtxt('asis_image\\female_names.txt', dtype=np.str_)
        self.__male_list = np.loadtxt('asis_image\\male_names.txt', dtype=np.str_)
        self.__male_list = self.__male_list[np.random.permutation(len(self.__male_list))[:len(self.__female_list)]]

    def _image_to_thumbnail(self):
        '''
            기존 원본 이미지를 특정 사이즈 형식으로 Thumbnail 을 수행하는 함수.
            이미지가 저장된 폴더로부터 이미지를 로드 후 썸네일 이미지 생성.
        '''
        print('Image Thumbnail 작업 시작')
        size = (126, 126)
        for label_list in zip(self.__male_list, self.__female_list):
            try:
                path_list = []
                for label in label_list:
                    for path in self.__image_path:
                        if label in path:
                            path_list.append(path)
                            break

                for idx, path in enumerate(path_list):
                    new_img = Image.new("RGB", size, "white")
                    im = Image.open(ImagePreprocessing.ASIS_IMAGE_PATH + 'lfw-deepfunneled\\' + path)
                    im.thumbnail(size, Image.ANTIALIAS)
                    load_img = im.load()
                    load_newimg = new_img.load()
                    i_offset = (size[0] - im.size[0]) / 2
                    j_offset = (size[1]- im.size[1]) / 2

                    for i in range(0, im.size[0]):
                        for j in range(0, im.size[1]):
                            load_newimg[i + i_offset, j + j_offset] = load_img[i, j]

                    new_img.save(ImagePreprocessing.ASIS_IMAGE_PATH + 'image_thumbnail\\' + label_list[idx])
            except Exception as e:
                print(e, label_list)
        print('Image Thumbnail 작업 완료')

    def _rgb2gray(self, rgb):
        '''
            YCrCb : 디지털(CRT, LCDl, PDP 등)을 위해서 따로 만들어둔 표현방법.
             - Y : Red*0.2126 + Green*0.7152 + Blue*0.0722
            YPbPr : 아날로그 시스템을 위한 표현방법.
             - Y : Red*0.299 + Green*0.587 + Blue*0.114
            실제 RGB 값들을 Gray Scale 로 변환하는 함수 .
        '''
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

        return np.array(gray).astype('int32')

    def _extract_rgb_from_image(self):
        '''
            썸네일 이미지로부터 RGB 데이터를 추출한 후 Gray 스케일로 변환하는 함수.
        '''
        print('RGB to Gray 작업 시작')
        for filename in os.listdir(ImagePreprocessing.ASIS_IMAGE_PATH + 'image_thumbnail\\'):
            try:
                img = mimage.imread(ImagePreprocessing.ASIS_IMAGE_PATH + 'image_thumbnail\\' + filename)
                gray = self._rgb2gray(img)
                if filename in self.__female_list:  # 여자 사진인 경우
                    self.__gray_data.append([gray, 0])
                else:
                    self.__gray_data.append([gray, 1])
            except OSError as e:
                print(str(filename) + ', 이미지를 식별할 수 없습니다.', e)
                continue

            self.__image_cnt += 1
            if self.__image_cnt % 1000 == 0:
                self._data_to_file()
                self.__gray_data.clear()

        self._data_to_file()
        self.__gray_data.clear()
        print('RGB to Gray 작업 완료')

    def _data_to_file(self):
        '''
            Gray Scale 로 변환된 이미지 정보를 파일로 기록하는 함수.
        '''
        print('데이터를 저장하는 중입니다.')
        for data in self.__gray_data:
            x_shape, y_shape = data[0].shape
            temp_data = ''
            for x in range(0, x_shape):
                for y in range(0, y_shape):
                    if x == 0 and y == 0:
                        temp_data += str(data[0][x][y])
                    else:
                        temp_data += ',' + str(data[0][x][y])
            temp_data += ',' + str(data[1])

            with open(ImagePreprocessing.TOBE_IMAGE_PATH + 'image_data_' + str(math.ceil(self.__image_cnt / 1000)) + '.csv', 'a', encoding='utf-8') as f:
                f.write(temp_data + '\n')
        print('데이터 저장이 완료되었습니다.')

    def play(self):
        '''
            이미지 전처리에 필요한 함수들을 수행.
        '''
        self._load_images()
        self._label_setting()
        self._image_to_thumbnail()
        self._extract_rgb_from_image()

preprocess = ImagePreprocessing()
preprocess.play()