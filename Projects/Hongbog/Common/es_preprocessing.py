import os
import shutil
from PIL import Image
import re
import numpy as np
import copy
import cv2

ORIGINAL_IMAGE_PATH = 'D:\\Data\\casia_original\\non-cropped'  # 분할 전 이미지 경로
NEW_IMAGE_PATH = 'D:\\Data\\casia_eyelid_segmentation\\use_data'  # 분할 후 이미지 경로
GROUND_TRUTH_PATH = os.path.join(NEW_IMAGE_PATH, 'ground_truth.txt')  # ground truth 에 대한 이미지 번호 정보가 있는 파일 경로

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
    path_list = get_file_path_list(ORIGINAL_IMAGE_PATH, [], 'F')

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
            try:
                for file_num in file_nums.split(','):
                    shutil.move(os.path.join(NEW_IMAGE_PATH, 'use_data', root_path, branch_path, leaf_path, 'cropped-image', file_num + extension),
                                os.path.join(NEW_IMAGE_PATH, 'use_data', root_path, branch_path, leaf_path, 'edge', file_num + extension))
                for file_path in os.listdir(os.path.join(NEW_IMAGE_PATH, 'use_data', root_path, branch_path, leaf_path, 'cropped-image')):
                    file_num = os.path.splitext(file_path)[0].split(os.path.sep)[-1]
                    shutil.move(os.path.join(NEW_IMAGE_PATH, 'use_data', root_path, branch_path, leaf_path, 'cropped-image', file_num + extension),
                                os.path.join(NEW_IMAGE_PATH, 'use_data', root_path, branch_path, leaf_path, 'non-edge', file_num + extension))
            except FileNotFoundError as ffe:
                print(ffe)

def image_gray_scale_extraction(img_path):
    '''
    경로에 해당하는 이미지 파일의 GrayScale 값을 추출하는 함수
    :param img_path: 이미지 경로
    :return: 이미지에서 추출된 GrayScale 값, type -> list
    '''
    img_data = []
    x_pixel, y_pixel = (16, 16)
    gray_img = Image.open(img_path).convert('L')  # RGB -> Image.open(img_path).convert('RGB')
    for y in range(0, y_pixel):
        for x in range(0, x_pixel):
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
        if re.search('.*non-edge.*', img_path):  # edge 이미지가 아닐경우 label -> 0로 설정
            img_gray_values.append(0)
        else:  # edge 이미지일 경우 label -> 1로 설정
            img_gray_values.append(1)
        img_data.append(img_gray_values)

        if len(img_data) == cnt_per_file:
            file_names = [int(os.path.splitext(file_name)[0]) for file_name in os.listdir(os.path.join(NEW_IMAGE_PATH, 'image_data'))]
            if len(file_names) == 0:  # 생성된 데이터 파일이 없는 경우
                data_to_file(img_data, 1)
            else:  # 기존에 생성된 데이터 파일이 있는 경우
                data_to_file(img_data, max(file_names)+1)
            img_data = []

'''
    Image Data 추가하는 방법
     1) ground_truth_image_move() : 추가되는 대상만 groud_truth 파일에 기술
     2) image_to_data()
'''

image_to_data()
# ground_truth_image_move()
# start_image_patching()

def connected_edge(predict):
    center_x, center_y = (3, 3)  # center location

    def init_data(predict):
        '''
        예측된 값에 대해 원하는 형태의 배열로 변환하는 함수
        :param predict: 인공 신경망에서 예측한 위치 번호
        :return: 실제 데이터(0, 1), 좌표 값 -> ndarray
        '''
        data = np.zeros((30, 40), dtype=np.int64)
        x = np.array(predict / 40, dtype=np.int32)
        y = np.array(predict % 40, dtype=np.int32)
        data[x, y] = 1
        loc = [(x, y) for x, y in zip(x, y)]
        return data, loc

    def print_data(data):
        '''
        데이터를 출력하는 함수
        :param data:
        :return:
        '''
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                print(' ' + str(data[x][y]), end='')
            print()
        print()

    def create_mask(loc):
        '''
        Edge 에 대한 각각의 mask 를 생성
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 1 1 1 1]
        [1 1 1 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        :param loc: 예측한 Edge 에 대한 좌표 값
        :return: 생성된 mask 값
        '''
        tot_mask = {}

        for x, y in loc:
            tmp_mask = np.zeros((7, 7))
            tmp_mask[-(x-3) if x-3 < 0 else 0:, -(y-3) if y-3 < 0 else 0:] = data[max(0, x-3):min(30, x+4), max(0, y-3):min(40, y+4)]
            tot_mask[(x, y)] = tmp_mask
        return tot_mask

    def search_conn_point(mask):
        '''
        다른 점들과 연결된 점을 찾는 함수
        (다른 점들과 연결이 되어 있으려면 해당 점을 기준으로 주위에 대칭으로 2개 이상의 점이 있다는 가정)
        :param mask:
        :return:
        '''
        def_mask = copy.deepcopy(mask[center_x - 1:center_x + 2, center_y - 1:center_y + 2])
        left_mask = def_mask[:, 0]
        right_mask = def_mask[:, 2]
        up_mask = def_mask[0, :]
        down_mask = def_mask[2, :]
        return (np.sum(left_mask) and np.sum(right_mask)) or (np.sum(up_mask) and np.sum(down_mask))
        # return True if np.sum(mask[center_x-1:center_x+2, center_y-1:center_y+2]) >= 3 else False

    def delete_mask(mask):
        del_keys = []

        for key in mask.keys():
            if search_conn_point(mask[key]):
                del_keys.append(key)

        for key in del_keys:
            mask.pop(key)

    def search_two_track(key, value):
        point = []
        condition = {(-2, -2): (-1, -1), (-1, -2): (-1, -1), (-2, -1): (-1, 0), (-2, 0): (-1, 0), (-2, 1): (-1, 0),
                     (-2, 2): (-1, 1), (-1, 2): (-1, 1), (0, 2): (0, 1), (1, 2): (1, 1), (2, 2): (1, 1), (2, 1): (1, 0),
                     (2, 0): (1, 0), (2, -1): (1, 0), (2, -2): (1, -1), (1, -2): (1, -1), (0, -2): (0, -1)}
        temp = copy.deepcopy(value)
        temp[center_x-1: center_x+2, center_y-1: center_y+2] = 0
        temp_x, temp_y = np.where(temp[center_x-2: center_x+3, center_y-2: center_y+3] == 1)

        for xx, yy in zip(temp_x, temp_y):
            con_x, con_y = condition[(xx-2, yy-2)]
            point.append((key[0] + con_x, key[1] + con_y))

        return tuple(point)

    def search_three_track(key, value):
        point = []
        condition = {(-3, -3): [(-1, -1), (-2, -2)], (-3, -2): [(-1, 0), (-2, -1)], (-3, -1): [(-1, 0), (-2, 0)], (-3, 0): [(-1, 0), (-2, 0)],
                     (-3, 1): [(-1, 0), (-2, 0)], (-3, 2): [(-1, 0), (-2, 1)], (-3, 3): [(-1, 1), (-2, 2)], (-2, 3): [(-1, 1), (-2, 2)],
                     (-1, 3): [(0, 1), (0, 2)], (0, 3): [(0, 1), (0, 2)], (1, 3): [(0, 1), (0, 2)], (2, 3): [(1, 1), (2, 2)],
                     (3, 3): [(1, 1), (2, 2)], (3, 2): [(1, 0), (2, 1)], (3, 1): [(1, 0), (2, 0)], (3, 0): [(1, 0), (2, 0)],
                     (3, -1): [(1, -1), (2, -1)], (3, -2): [(1, -1), (2, -2)], (3, -3): [(1, -1), (2, -2)], (2, -3): [(1, -1), (2, -2)],
                     (1, -3): [(1, -1), (1, -2)], (0, -3): [(0, -1), (0, -2)], (-1, -3): [(-1, -1), (-1, -2)], (-2, -3): [(-1, -1), (-2, -2)]}
        temp = copy.deepcopy(value)
        temp[center_x-2: center_x+3, center_y-2: center_y+3] = 0
        temp_x, temp_y = np.where(temp == 1)

        for xx, yy in zip(temp_x, temp_y):
            for con_x, con_y in condition[(xx-3, yy-3)]:
                point.append((key[0] + con_x, key[1] + con_y))

        return tuple(point)

    '''data initialization'''
    data, loc = init_data(predict)
    print_data(data)

    masks = create_mask(loc)
    delete_mask(masks)  # 15 개

    '''two track'''
    points = set()
    for key in masks.keys():
        for point in search_two_track(key, masks[key]):
            points.add(point)

    point_x, point_y = [], []

    for px, py in points:
        point_x.append(px)
        point_y.append(py)

    data[point_x, point_y] = 1
    print_data(data)

    '''three track'''
    delete_mask(masks)

    points = set()
    for key in masks.keys():
        for point in search_three_track(key, masks[key]):
            points.add(point)

    point_x, point_y = [], []

    for x, y in points:
        point_x.append(x)
        point_y.append(y)

    data[point_x, point_y] = 1
    print_data(data)

    return data

def point_count(edge_data):
    edge_data_bak = copy.deepcopy(edge_data)

    def print_data(data):
        '''
        데이터를 출력하는 함수
        :param data:
        :return:
        '''
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                print(' ' + str(data[x][y]), end='')
            print()
        print()

    def remove_loc():
        edge_data_bak[:2, :] = 0
        edge_data_bak[-2:, :] = 0

    def convert_to_loc():
        loc_data = []
        for x in range(edge_data_bak.shape[0]):
            for y in range(edge_data_bak.shape[1]):
                if edge_data_bak[x, y] == 1:
                    loc_data.append((x, y))

        return loc_data

    def point_to_point(loc_data):
        point_cnt = np.zeros((30, 40), dtype=np.int64)

        for loc in loc_data:
            sub_loc_data = copy.deepcopy(loc_data)
            sub_loc_data.remove(loc)
            for sub_loc in sub_loc_data:
                temp_points = set()
                for x in range(loc[0]+1 if loc[0] <= sub_loc[0] else sub_loc[0]+1, sub_loc[0] if loc[0] <= sub_loc[0] else loc[0]):
                    y = int((sub_loc[1]-loc[1])/(sub_loc[0]-loc[0])*(x-loc[0])+loc[1])
                    temp_points.add((x, y))

                for y in range(loc[1]+1 if loc[1] <= sub_loc[1] else sub_loc[1]+1, sub_loc[1] if loc[1] <= sub_loc[1] else loc[1]):
                    x = int((sub_loc[0]-loc[0])/(sub_loc[1]-loc[1])*(y-loc[1])+loc[0])
                    temp_points.add((x, y))

                for temp_point in temp_points:
                    x, y = temp_point
                    point_cnt[x, y] = point_cnt[x, y] + 1

        return point_cnt

    remove_loc()
    loc_data = convert_to_loc()
    point_cnt = point_to_point(loc_data)
    print_data(point_cnt)
    print(np.where(point_cnt == np.max(point_cnt)))


# predict = np.array([416, 417, 418, 419, 420, 452, 453, 454, 455, 462, 463, 464, 465, 466, 467, 468, 490, 491, 509, 510, 511, 512, 529, 554, 555, 568, 607, 633, 671, 672, 687, 688, 690, 691, 692, 710, 733, 734, 735, 736, 737, 739, 744, 746, 747, 779, 780, 781, 783, 784])
# predict = np.array([8, 11, 13, 14, 17, 20, 26, 30, 51, 52, 56, 59, 91, 94, 377, 420, 453, 454, 459, 460, 461, 462, 466, 491, 504, 505, 506, 530, 532, 547, 548, 585, 589, 591, 671, 692, 733, 753, 774, 775, 816, 817, 835, 855, 860, 861, 865, 904, 908, 909])
# predict = np.array([382, 383, 384, 417, 418, 419, 420, 421, 422, 423, 424, 425, 427, 428, 454, 455, 456, 457, 465, 466, 467, 469, 470, 493, 494, 496, 511, 512, 531, 532, 553, 570, 571, 594, 609, 648, 687, 692, 711, 730, 731, 748, 749, 750, 767, 768, 769, 772, 773, 774, 775, 776, 777, 778, 779, 783, 784, 785, 786, 787, 818, 819, 820, 821, 1188])
# predict = np.array([68, 380, 381, 382, 383, 384, 385, 386, 416, 417, 418, 419, 426, 427, 428, 429, 430, 454, 455, 468, 471, 472, 492, 493, 514, 531, 572, 649, 651, 652, 653, 654, 655, 672, 696, 697, 698, 699, 700, 706, 708, 709, 710, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748])
# predict = np.array([15, 16, 66, 87, 113, 129, 333, 339, 340, 341, 371, 372, 377, 378, 379, 380, 382, 385, 414, 420, 427, 450, 510, 530, 571, 633, 634, 650, 652, 653, 654, 673, 674, 690, 695, 696, 697, 705, 706, 707, 708, 709, 710, 737, 738, 739, 740, 741, 742, 743, 744, 745, 750])
# edge_data = connected_edge(predict)
# point_count(edge_data)


'''
new_img = Image.new('L', (640, 480))
x_pixel, y_pixel = new_img.size
x_delta, y_delta, img_cnt = int(x_pixel / 40), int(y_pixel / 30), 1
predict_edges2 = [8, 11, 13, 14, 17, 20, 26, 30, 51, 52, 56, 59, 91, 94, 377, 420, 453, 454, 459, 460, 461, 462, 466, 491, 504, 505, 506, 530, 532, 547, 548, 585, 589, 591, 671, 692, 733, 753, 774, 775, 816, 817, 835, 855, 860, 861, 865, 904, 908, 909]
# predict_edges1 = [15, 16, 66, 87, 113, 129, 333, 339, 340, 341, 371, 372, 377, 378, 379, 380, 382, 385, 414, 420, 427, 450, 510, 530, 571, 633, 634, 650, 652, 653, 654, 673, 674, 690, 695, 696, 697, 705, 706, 707, 708, 709, 710, 737, 738, 739, 740, 741, 742, 743, 744, 745, 750]
# predict_edges = [416, 417, 418, 419, 420, 452, 453, 454, 455, 462, 463, 464, 465, 466, 467, 468, 490, 491, 509, 510, 511, 512, 529, 554, 555, 568, 607, 633, 671, 672, 687, 688, 690, 691, 692, 710, 733, 734, 735, 736, 737, 739, 744, 746, 747, 779, 780, 781, 783, 784]

for init_y in range(0, y_pixel, y_delta):
    for init_x in range(0, x_pixel, x_delta):
        if img_cnt in predict_edges2:
            for y in range(init_y, init_y + y_delta):
                for x in range(init_x, init_x + x_delta):
                    new_img.putpixel((x, y), 255)
        img_cnt = img_cnt + 1

# for x in range(30):
#     for y in range(40):
#         if output[x][y] == 1:
#             new_img.putpixel((y, x), 255)
#         else:
#             new_img.putpixel((y, x), 0)
new_img.save('D:\\OneDrive\\00.신경망\\Eyelid Segmentation\\테스트 결과\\connected edge\\new_file2.jpg')
# new_img.show()
'''