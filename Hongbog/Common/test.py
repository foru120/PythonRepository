import shutil
import os

asis_path = 'D:\\100_dataset\\casia_blurring\\image_data'
tobe_path = 'D:\\100_dataset\\casia_blurring\\backup'

file_num = sorted([num + n for n in range(11, 100) for num in range(0, 9597, 100)])

for num in file_num:
    try:
        shutil.move(os.path.join(asis_path, str(num) + '.txt'), os.path.join(tobe_path, str(num) + '.txt'))
    except FileNotFoundError:
        print(str(num) + '.txt, 파일이 존재하지 않습니다.')