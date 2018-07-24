import cv2
import os
import numpy as np
import re

# img directory에서 Xdata, Ydata를 각각 read 할 수 있게 경로별로 파일명을 만드는 코드
img_dir = "D:\\artery_labeled\\png\\"
p = re.compile('D:\\\\artery_labeled\\\\png\\\\([^\\\\.]*)')
data_paths = {dirname: [] for dirname in os.listdir(img_dir)}

def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for idx, filename in enumerate(filenames):
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else:
                if idx >= 140:
                    return
                key = p.search(full_filename).group(1)
                temp = data_paths[key]
                if dirname.endswith('X'):
                    temp.append([full_filename])
                elif dirname.endswith('Y'):
                    temp[idx].append(full_filename)
    except PermissionError:
        pass

def data_setting():  # [[x, y]....]
    total_data_x = []
    total_data_y = []
    for key in data_paths.keys():
        patient_data_x = []
        patient_data_y = []
        for data in data_paths[key]:
            img_x = cv2.imread(data[0])
            img_y = cv2.imread(data[1])

            gray_image_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
            gray_image_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2GRAY)

            # random crop
            width = np.random.randint(0, gray_image_x.shape[0] - 224)
            height = np.random.randint(0, gray_image_x.shape[1] - 224)
            gray_image_x = np.reshape(gray_image_x[width: width+224, height: height+224], (1, 224, 224))

            patient_data_x.append(gray_image_x.tolist())
            patient_data_y.append(gray_image_y.tolist())
        total_data_x.append(patient_data_x)
    return np.array(total_data_x), np.array(total_data_y)

search(img_dir)
print(data_setting()[0].shape, data_setting()[1].shape)
