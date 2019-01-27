import os, shutil

#todo 원본 데이터 셋 위치
original_dataset_dir = '/home/kyh/dataset/dogs_and_cats/train'

#todo 데이터 셋 분할 위치 폴더 생성
base_dir = '/home/kyh/dataset/dogs_and_cats_small'
os.mkdir(base_dir)

#todo 훈련/검증/테스트 데이터 셋 폴더 생성
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

#todo 각 데이터 셋 별 이미지 복사
dst_cats_dir = [train_cats_dir, validation_cats_dir, test_cats_dir]
dst_dogs_dir = [train_dogs_dir, validation_dogs_dir, test_dogs_dir]

def file_copy(src, dst):
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(2000)]
fname_list = [fnames[:1000], fnames[1000:1500], fnames[1500:]]
dst_list = [train_cats_dir, validation_cats_dir, test_cats_dir]

for idx in range(len(fname_list)):
    for fname in fname_list[idx]:
        file_copy(src=os.path.join(original_dataset_dir, fname),
                  dst=os.path.join(dst_list[idx], fname))

fnames = ['dog.{}.jpg'.format(i) for i in range(2000)]
fname_list = [fnames[:1000], fnames[1000:1500], fnames[1500:]]
dst_list = [train_dogs_dir, validation_dogs_dir, test_dogs_dir]

for idx in range(len(fname_list)):
    for fname in fname_list[idx]:
        file_copy(src=os.path.join(original_dataset_dir, fname),
                  dst=os.path.join(dst_list[idx], fname))

print('훈련용 고양이 이미지 전체 개수:', len(os.listdir(train_cats_dir)))
print('훈련용 강아지 이미지 전체 개수:', len(os.listdir(train_dogs_dir)))
print('검증용 고양이 이미지 전체 개수:', len(os.listdir(validation_cats_dir)))
print('검증용 강아지 이미지 전체 개수:', len(os.listdir(validation_dogs_dir)))
print('테스트용 고양이 이미지 전체 개수:', len(os.listdir(test_cats_dir)))
print('테스트용 강아지 이미지 전체 개수:', len(os.listdir(test_dogs_dir)))