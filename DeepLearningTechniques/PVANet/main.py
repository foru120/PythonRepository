import json
from collections import OrderedDict
from pprint import pprint

'''COCO 2017 Dataset '''
TRAIN_FILE_PATH = 'D:\\100_dataset\\coco\\train\\train2017\\'
VALIDATION_FILE_PATH = 'D:\100_dataset\coco\validation\val2017\\'
TEST_FILE_PATH = 'D:\100_dataset\coco\test\test2017\\'
ANNOTATION_FILE_PATH = 'D:\\100_dataset\\coco\\annotations_trainval2017\\annotations\\'

'''json 형식의 파일 내용을 불러오는 함수'''
def load_data_from_json():
    with open(ANNOTATION_FILE_PATH+'instances_train2017.json') as file:
        data = json.load(file, object_pairs_hook=OrderedDict)
    pprint(data.keys())

load_data_from_json()