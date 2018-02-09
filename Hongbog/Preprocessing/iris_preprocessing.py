import os
import shutil
from PIL import Image
import re
import numpy as np

ORIGINAL_IMAGE_PATH = 'D:\\Data\\CASIA\\CASIA-IrisV2\\CASIA-IrisV2'  # 분할 전 이미지 경로
NEW_IMAGE_PATH = 'D:\\Data\\casia_preprocessing'  # 분할 후 이미지 경로
GROUND_TRUTH_PATH = os.path.join(NEW_IMAGE_PATH, 'ground_truth.txt')  # ground truth 에 대한 이미지 번호 정보가 있는 파일 경로

def get_file_path_list(root_path, path_list, mode):
    '''
    특정 폴더 밑에 있는 파일 경로를 리스트화하는 함수 (os.walk() 참조)
    :param root_path: 파일이 포함된 상위 경로
    :param path_list: 파일의 경로 정보를 담을 리스트
    :param mode: 'C' -> (Cropped) Cropped 된 이미지 폴더를 대상으로 파일 검출,  'E' -> (Edge), edge 와 non-edge 폴더를 대상으로 파일 검출
    :return: 이미지 경로, type -> list
    '''
    for leaf_path in os.listdir(root_path):
        full_path = os.path.join(root_path, leaf_path)
        if os.path.isdir(full_path):
            path_list = get_file_path_list(full_path, path_list, mode)
        elif os.path.isfile(full_path):
            if mode == 'C':
                path_list.append(full_path)
            elif mode == 'E':
                if re.search('.*(edge)|(non-edge).*', full_path):
                    path_list.append(full_path)
    return path_list

def create_directory(root_folder, branch_folder, file_name):
    '''
    디렉토리를 생성하는 함수
    :param root_folder: 생성할 디렉토리의 최상위 경로 (device1, device2)
    :param branch_folder: 생성할 디렉토리의 중간 분기 경로 (0000, 0001, 0002...)
    :param file_name: 생성할 디렉토리의 상위 경로 (0000_000, 0000_001, 0000_002...)
    :return: None
    '''
    leaf_folder = ['edge', 'non-edge', 'cropped-image']

    for folder in leaf_folder:
        os.makedirs(os.path.join(NEW_IMAGE_PATH, root_folder, branch_folder, file_name, folder))

def create_patch_image(root_folder, branch_folder, file_name, ori_path):
    '''
    패치 사이즈 단위로 이미지 잘라내는 함수
    :param root_folder: 새로 생성된 패치 이미지 디렉토리의 최 상위 경로
    :param branch_folder: 새로 생성된 패치 이미지 디렉토리의 중간 분기 경로
    :param file_name: 새로 생성된 패치 이미지 디렉토리의 상위 경로
    :param ori_path: 원본 이미지 경로
    :return: None
    '''
    x_pixel, y_pixel = 640, 480  # x_pixel, y_pixel: 입력 이미지 x축, y축 픽셀 값
    x_delta, y_delta, img_cnt = 16, 16, 1  # x_delta, y_delta: 이미지 x축, y축 분할 단위, img_cnt: 이미지 순번을 위한 변수

    ori_img = Image.open(ori_path)  # rgb_im = im.convert('RGB')
    width, height = ori_img.size

    if (width == x_pixel) and (height == y_pixel):
        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                new_img = Image.new('L', (x_delta, y_delta))  # GrayScale 이미지로 생성, Image.new('RGB', (width, height)): RGB 이미지로 생성
                for y in range(init_y, init_y + y_delta):
                    for x in range(init_x, init_x + x_delta):
                        new_img.putpixel((x % x_delta, y % y_delta), ori_img.getpixel((x, y)))
                new_img.save(os.path.join(NEW_IMAGE_PATH, root_folder, branch_folder, file_name, 'cropped-image', str(img_cnt)+'.bmp'))
                img_cnt = img_cnt + 1

def start_image_patching():
    '''
    이미지 패치 관련 Main 함수
    :return: None
    '''
    path_list = get_file_path_list(ORIGINAL_IMAGE_PATH, [], 'C')
    for ori_path in path_list:
        root_folder, branch_folder, file_name = os.path.splitext(ori_path)[0].split(os.path.sep)[-3: ]
        if not os.path.isdir(os.path.join(NEW_IMAGE_PATH, root_folder, branch_folder, file_name)):
            create_directory(root_folder, branch_folder, file_name)
        print('create patch image, ', root_folder, branch_folder, os.path.splitext(file_name)[0])
        create_patch_image(root_folder, branch_folder, file_name, ori_path)

def ground_truth_image_move():
    '''
    edge 이미지와 non-edge 이미지를 각각 해당 폴더로 이동시키는 함수
    :return: None
    '''
    extension = '.bmp'  # 확장자

    with open(GROUND_TRUTH_PATH) as file:
        for text in file:
            root_path, branch_path, leaf_path, file_nums = text.split()
            for file_num in file_nums.split(','):
                shutil.move(os.path.join(NEW_IMAGE_PATH, root_path, branch_path, leaf_path, 'cropped-image', file_num + extension),
                            os.path.join(NEW_IMAGE_PATH, root_path, branch_path, leaf_path, 'edge', file_num + extension))
            for file_path in os.listdir(os.path.join(NEW_IMAGE_PATH, root_path, branch_path, leaf_path, 'cropped-image')):
                file_num = os.path.splitext(file_path)[0].split(os.path.sep)[-1]
                shutil.move(os.path.join(NEW_IMAGE_PATH, root_path, branch_path, leaf_path, 'cropped-image', file_num + extension),
                            os.path.join(NEW_IMAGE_PATH, root_path, branch_path, leaf_path, 'non-edge', file_num + extension))

def image_gray_scale_extraction(img_path):
    '''
    경로에 해당하는 이미지 파일의 GrayScale 값을 추출하는 함수
    :param img_path: 이미지 경로
    :return: 이미지에서 추출된 GrayScale 값, type -> list
    '''
    img_data = []
    x_pixel, y_pixel = (16, 16)
    gray_img = Image.open(img_path).convert('L')  # RGB -> Image.open(img_path).convert('RGB')
    for x in range(0, x_pixel):
        for y in range(0, y_pixel):
            img_data.append(gray_img.getpixel((x, y)))
    return img_data

def data_to_file(img_data, name):
    '''
    추출된 이미지 값을 file 로 저장하는 함수
    :param img_data: 추출된 이미지 값
    :param name: 파일 이름
    :return: None
    '''
    with open(os.path.join(NEW_IMAGE_PATH, 'image_data', str(name)+'.txt'), mode='w') as file:
        for idx in np.random.permutation(len(img_data)):
            file.write(','.join([str(data) for data in img_data[idx]]) + '\n')

def image_to_data():
    '''
    만들어진 패치 크기 이미지의 Gray Scale 값을 파일로 저장하는 함수
    :return: None
    '''
    img_data, cnt_per_file = [], 1200
    for img_path in get_file_path_list(NEW_IMAGE_PATH, [], 'E'):
        img_gray_values = image_gray_scale_extraction(img_path)
        if re.search('.*non-edge.*', img_path):  # edge 이미지가 아닐경우 label -> 1로 설정
            img_gray_values.append(0)
        else:  # edge 이미지일 경우 label -> 2로 설정
            img_gray_values.append(1)
        img_data.append(img_gray_values)

        if len(img_data) == cnt_per_file:
            file_names = [int(os.path.splitext(file_name)[0]) for file_name in os.listdir(os.path.join(NEW_IMAGE_PATH, 'image_data'))]
            if len(file_names) == 0:  # 생성된 데이터 파일이 없는 경우
                data_to_file(img_data, 1)
            else:  # 기존에 생성된 데이터 파일이 있는 경우
                data_to_file(img_data, max(file_names)+1)
            img_data = []

image_to_data()