import os
from PIL import Image

# ORIGINAL_IMAGE_PATH = 'D:\\100_dataset\\iris\\CASIA\\CASIA-IrisV2\\CASIA-IrisV2'  # 분할 전 이미지 경로
ORIGINAL_IMAGE_PATH = 'D:\\Data\\CASIA\\CASIA-IrisV2\\CASIA-IrisV2'
NEW_IMAGE_PATH = 'D:\\100_dataset\\iris\\casia_preprocessing'  # 분할 후 이미지 경로

def get_file_path_list(root_path, path_list):
    for leaf_path in os.listdir(root_path):
        full_path = os.path.join(root_path, leaf_path)
        if os.path.isdir(full_path):
            path_list = get_file_path_list(full_path, path_list)
        elif os.path.isfile(full_path):
            path_list.append(full_path)
    return path_list

def create_directory(root_folder, branch_folder, file_name):
    leaf_folder = ['edge', 'non-edge', 'cropped-image']

    for folder in leaf_folder:
        os.makedirs(os.path.join(NEW_IMAGE_PATH, root_folder, branch_folder, file_name, folder))

def create_patch_image(root_folder, branch_folder, file_name, ori_path):
    x_pixel, y_pixel = 640, 480  # x_pixel, y_pixel: 입력 이미지 x축, y축 픽셀 값
    x_delta, y_delta, img_cnt = 16, 16, 1  # x_delta, y_delta: 이미지 x축, y축 분할 단위, img_cnt: 이미지 순번을 위한 변수

    ori_img = Image.open(ori_path)  # rgb_im = im.convert('RGB')
    width, height = ori_img.size

    if (width == x_pixel) and (height == y_pixel):
        for init_y in range(0, y_pixel, y_delta):
            for init_x in range(0, x_pixel, x_delta):
                new_img = Image.new('L', (x_delta, y_delta))  # GRAY 이미지로 생성, Image.new('RGB', (width, height)): RGB 이미지로 생성
                for y in range(init_y, init_y + y_delta):
                    for x in range(init_x, init_x + x_delta):
                        new_img.putpixel((x % x_delta, y % y_delta), ori_img.getpixel((x, y)))
                new_img.save(os.path.join(NEW_IMAGE_PATH, root_folder, branch_folder, file_name, 'cropped-image', str(img_cnt)+'.bmp'))
                img_cnt = img_cnt + 1

def start_image_patching():
    path_list = get_file_path_list(os.path.abspath(ORIGINAL_IMAGE_PATH), [])
    for ori_path in path_list:
        root_folder, branch_folder, file_name = os.path.splitext(ori_path)[0].split(os.path.sep)[-3: ]
        if not os.path.isdir(os.path.join(NEW_IMAGE_PATH, root_folder, branch_folder, file_name)):
            create_directory(root_folder, branch_folder, file_name)
        print('create patch image, ', root_folder, branch_folder, os.path.splitext(file_name)[0])
        create_patch_image(root_folder, branch_folder, file_name, ori_path)

import re
path_list = get_file_path_list(os.path.abspath(ORIGINAL_IMAGE_PATH), [])
for ori_path in path_list:
    root_folder, branch_folder, file_name = os.path.splitext(ori_path)[0].split(os.path.sep)[-3:]
    if re.match('.*\.bmp', ori_path):
        ori_img = Image.open(ori_path)  # rgb_im = im.convert('RGB')
        width, height = ori_img.size
        if (width == 640) and (height == 480):
            print('right' if root_folder == 'device1' else 'left', file_name)
# start_image_patching()