import os
import re
from PIL import Image
import cv2
import numpy as np
from collections import defaultdict

ORI_IMAGE_PATH = 'D:\\Data\\casia_original'
BLUR_IMAGE_PATH = 'D:\\Data\\casia_blurring'

def image_blurring():
    asis_path = 'D:\\Data\\CASIA\\CASIA-IrisV2\\CASIA-IrisV2'
    tobe_path = 'D:\\Data\\casia_blurring\\non-cropped'

    for img_path in get_file_path_list(asis_path, [], 'F'):
        folder_name = re.match('\\\(.*)\\\(.*\.bmp)', img_path[img_path.index(asis_path)+len(asis_path):])
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

def get_file_path_list(root_path, path_list, mode):
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
            path_list = get_file_path_list(full_path, path_list, mode)
        elif os.path.isfile(full_path):
            if mode == 'F':
                path_list.append(full_path)
            elif mode == 'E':
                if re.search('.*(edge)|(non-edge).*', full_path):
                    path_list.append(full_path)
            elif mode == 'C':
                if re.search('.*\\\cropped\\\.*', full_path):
                    path_list.append(full_path)
    return path_list

def image_gray_scale_extraction(img_path):
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

def data_to_file(data, name):
    '''
    추출된 이미지 값을 file 로 저장하는 함수
    :param img_data: 추출된 이미지 값
    :param name: 파일 이름
    :return: None
    '''
    with open(os.path.join(BLUR_IMAGE_PATH, 'image_data', str(name)+'.txt'), mode='w') as file:
        for idx in np.random.permutation(len(data)):
            file.write(','.join([str(d) for d in data[idx][0]]) + ','.join([str(d) for d in data[idx][1]]) + '\n')

def image_to_data():
    '''
    만들어진 패치 크기 이미지의 Gray Scale 값을 파일로 저장하는 함수
    :return: None
    '''
    ori_img_data, cnt_per_batch, file_cnt = defaultdict(dict), 100, 1
    blur_sort = ['bilateral', 'blur', 'gaussian', 'median']

    for idx, ori_img_path in enumerate(get_file_path_list(os.path.join(ORI_IMAGE_PATH, 'cropped'), [], 'F')):
        ori_img_gray_values = image_gray_scale_extraction(ori_img_path)
        m = re.match('(.*)\\\(.*)\\\(.*)\\\(.*)\\\(.*)\\\(.*)\\\(.*)\\\(.*)\.bmp', ori_img_path).groups()
        path1, path2, path3, ori_filename = m[4:]
        ori_img_data[path1 + ',' + path2 + ',' + path3][ori_filename] = ori_img_gray_values

        if (idx+1)%cnt_per_batch == 0:
            tot_image_data = []
            delete_keys = []
            for key in ori_img_data.keys():
                if len(ori_img_data[key].keys()) == 100:
                    delete_keys.append(key)

            for key in delete_keys:
                blur_img_paths = []
                for sort in blur_sort:
                    p1, p2, p3 = key.split(',')
                    blur_img_paths.append(os.path.join(BLUR_IMAGE_PATH, 'cropped', p1, p2, p3 + '_' + sort))

                for blur_img_path in blur_img_paths:
                    for path in get_file_path_list(blur_img_path, [], 'F'):
                        blur_filename = re.match('.*\\\(.*)\.bmp', path).groups()[0]
                        blur_img_gray_values = image_gray_scale_extraction(path)
                        tot_image_data.append([blur_img_gray_values, ori_img_data[key][blur_filename]])

                ori_img_data.pop(key)

            for idx in range(0, len(tot_image_data), 100):
                print('create file, ' + str(file_cnt) + '.txt')
                data_to_file(tot_image_data[idx:idx+100], file_cnt)
                file_cnt = file_cnt + 1

image_to_data()