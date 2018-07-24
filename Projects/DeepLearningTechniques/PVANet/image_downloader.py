import os
import re
import urllib.request

FILE_PATH = 'D:\\100_dataset\\imagenet\\imagenet_fall11_urls\\fall11.txt'
DIR_PATH = 'D:\\100_dataset\\imagenet\\images\\'

'''다운로드할 imageID 와 url 정보를 가진 text 파일 파싱'''
def read_file():
    cnt = 0
    with open(FILE_PATH, mode='rt', encoding='utf-8', errors='ignore') as file:
        for line in file:
            m = re.match('(.+)\t(.+)\n', line)
            if m:
                imageID, url = m.groups()
                worldID = imageID[:imageID.index('_')]
                is_directory(worldID)
                is_down = image_download(worldID, imageID, url)
                if is_down:
                    cnt += 1
                    if cnt % 100 == 0:
                        print(str(cnt) + ' 개, downloading...')
    print('총 ' + str(cnt) + ' 개 다운로드 하였습니다.')

'''디렉토리가 존재하는지 확인'''
def is_directory(worldID):
    if not os.path.isdir(DIR_PATH + worldID):
        os.mkdir(DIR_PATH + worldID)

'''주어진 url 로 이미지 다운로드'''
def image_download(worldID, imageID, url):
    try:
        urllib.request.urlretrieve(url, DIR_PATH + worldID + '\\' + imageID + '.jpg')
        return True
    except Exception as e:
        print(url, e)
        return False

'''특정 파일 삭제'''
def file_remove(worldID, imageID):
    try:
        os.remove(DIR_PATH + worldID + '\\' + imageID + '.jpg')
    except Exception as e:
        print(imageID, e)

read_file()