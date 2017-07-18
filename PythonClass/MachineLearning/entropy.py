import math
import csv
import time
import psutil, os
# dataset = [({'cust_name':'SCOTT', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
#            ({'cust_name':'SMITH', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
#            ({'cust_name':'ALLEN', 'card_yn':'N', 'review_yn':'N', 'before_buy_yn':'Y'}, False),
#            ({'cust_name':'JONES', 'card_yn':'Y', 'review_yn':'N', 'before_buy_yn':'N'}, True),
#            ({'cust_name':'WARD',  'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True)]

# 시작 메모리 체크
proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]

dataset = []

def returnCsvData(filename):
    file = open('D:\\KYH\\02.PYTHON\\data\\'+filename, 'r', encoding='utf-8')
    return csv.DictReader(file)

def entropy(data):
    tot = sum([sum(d) for d in data.values()])
    return sum((sum(d) / tot) *
               (-((d[0] / sum(d)) * math.log2(d[0] / sum(d)) if d[0] != 0 else 0)
                - ((d[1] / sum(d)) * math.log2(d[1] / sum(d)) if d[1] != 0 else 0)) for d in data.values())

def setting_data(dataset):
    groups = {}
    for input in dataset:
        for key, value in input[0].items():
            if groups.get(key) is None:
                groups[key] = {}
            if groups[key].get(value) is None:
                groups[key][value] = [0, 0]
            groups[key][value][input[1]] += 1
    return groups

start_time = time.time()
for data in returnCsvData('Cosmetics_decisionTree.csv'):
    *keys, base_key = data.keys()
    dataset.append(({key: data[key] for key in keys}, False if data[base_key] == '0' else True))

print([(key, entropy(value)) for key, value in setting_data(dataset).items()])
end_time = time.time()
print(end_time-start_time)

# 실행 후 맨 밑에서 코드 구동 후 메모리 체크
proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)