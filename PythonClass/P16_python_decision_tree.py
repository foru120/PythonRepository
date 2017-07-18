print('')
print('====================================================================================================')
print('== 문제 344. (점심시간) 확률 0.4 의 엔트로피 지수를 구하시오!')
print('====================================================================================================')
import math
print(-0.4*math.log2(0.4))


print('')
print('====================================================================================================')
print('== 문제 345. 카드 유무의 컬럼의 데이터의 분할 전 엔트로피를 구하시오.')
print('====================================================================================================')
import math
print(-(4/5)*math.log2((4/5)) - (1/5)*math.log2((1/5)))


print('')
print('====================================================================================================')
print('== 문제 346. 확률을 입력했을 때 엔트로피를 출력하는 파이썬 함수를 생성하시오.')
print('====================================================================================================')
def entropy(p_list):
    return sum(-(p)*math.log2(p) for p in p_list)

print(entropy([4/5, 1/5]))


print('')
print('====================================================================================================')
print('== 문제 347. 아래의 리스트 변수에 있는 y와 n이 y가 4개 있고 n이 1개 있다는게 자동으로 구분되어지게 하려면?')
print('====================================================================================================')
import math
import collections
card_yn = ['Y', 'Y', 'N', 'Y', 'Y', 'K', 'K']

def class_probabilities(labels):
    return collections.Counter(labels).values()

print(class_probabilities(card_yn))


print('')
print('====================================================================================================')
print('== 문제 348. 위의 코드를 이용해서 아래의 결과가 출력될수 있도록 하시오!')
print('====================================================================================================')
card_yn = ['Y', 'Y', 'N', 'Y', 'Y']

def class_probabilities(labels):
    return [value/len(labels) for value in collections.Counter(labels).values()]

print(class_probabilities(card_yn))


print('')
print('====================================================================================================')
print('== 문제 349. 위에서 만든 함수 2개로 엔트로피를 구하시오!')
print('====================================================================================================')
print(entropy(class_probabilities(card_yn)))


print('')
print('====================================================================================================')
print('== 문제 352. 아래의 데이터 셋에서 scott 을 출력하려면?')
print('====================================================================================================')
inputs = [{'ename':'scott'}, True]
print(inputs[0]['ename'])


print('')
print('====================================================================================================')
print('== 문제 353. 아래의 결과에서 scott 과 smith 를 출력하려면?')
print('====================================================================================================')
inputs = [({'ename':'scott'}, True), ({'ename':'smith'}, False)]
print(','.join(v[0]['ename'] for v in inputs))


print('')
print('====================================================================================================')
print('== 문제 354. 아래의 데이터 셋에서 card_yn 의 y와 n만 출력하려면?')
print('====================================================================================================')
inputs = [({'ename': 'scott', 'card_yn': 'y'}, True), ({'ename': 'smith', 'card_yn': 'n'}, False)]
print(','.join(v[0]['card_yn'] for v in inputs))


print('')
print('====================================================================================================')
print('== 문제 355. 아래의 inputs 데이터셋에서 card_yn 의 y 와 n 을 groups 라는 비어있는 리스트 변수에 넣으시오!')
print('====================================================================================================')
inputs = [({'cust_name':'SCOTT', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ({'cust_name':'SMITH', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ({'cust_name':'ALLEN', 'card_yn':'N', 'review_yn':'N', 'before_buy_yn':'Y'}, False),
          ({'cust_name':'JONES', 'card_yn':'Y', 'review_yn':'N', 'before_buy_yn':'N'}, True),
          ({'cust_name':'WARD',  'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True)]

groups = []
for v in inputs:
    groups.append(v[0]['card_yn'])
print(groups)


print('')
print('====================================================================================================')
print('== 문제 356. 위의 코드를 가지고 아래의 결과가 출력되게 하시오!')
print('====================================================================================================')
inputs = [({'cust_name':'SCOTT', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ({'cust_name':'SMITH', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ({'cust_name':'ALLEN', 'card_yn':'N', 'review_yn':'N', 'before_buy_yn':'Y'}, False),
          ({'cust_name':'JONES', 'card_yn':'Y', 'review_yn':'N', 'before_buy_yn':'N'}, True),
          ({'cust_name':'WARD',  'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True)]

def setting_data(inputs):
    groups = {}
    for input in inputs:
        for key, value in input[0].items():
            if key != 'cust_name':
                if groups.get(key) is None:
                    groups[key] = []
                groups[key].append(value)
    return groups

print(setting_data(inputs))


print('')
print('====================================================================================================')
print('== 문제 357. 아래와 같이 분할전 엔트로피가 출력되게 위의 코드를 수정하시오.')
print('====================================================================================================')
import math
import collections

inputs = [({'cust_name':'SCOTT', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ({'cust_name':'SMITH', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
          ({'cust_name':'ALLEN', 'card_yn':'N', 'review_yn':'N', 'before_buy_yn':'Y'}, False),
          ({'cust_name':'JONES', 'card_yn':'Y', 'review_yn':'N', 'before_buy_yn':'N'}, True),
          ({'cust_name':'WARD',  'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True)]

def entropy(p_list):
    return sum(-(p)*math.log2(p)-(1-p)*math.log2(1-p) for p in p_list)

def class_probabilities(labels):
    return [value/len(labels) for value in collections.Counter(labels).values()]

def setting_data(inputs):
    groups = {}
    for input in inputs:
        for key, value in input[0].items():
            if key != 'cust_name':
                if groups.get(key) is None:
                    groups[key] = []
                groups[key].append(value)
    return groups

print([(key, entropy(class_probabilities(value))) for key, value in setting_data(inputs).items()])


print('')
print('====================================================================================================')
print('== 문제 362. 구매여부 데이터에 대한 분할 후 엔트로피를 출력하는 코드를 수행하시오')
print('====================================================================================================')
import math
import csv
import time


# dataset = [({'cust_name':'SCOTT', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
#            ({'cust_name':'SMITH', 'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True),
#            ({'cust_name':'ALLEN', 'card_yn':'N', 'review_yn':'N', 'before_buy_yn':'Y'}, False),
#            ({'cust_name':'JONES', 'card_yn':'Y', 'review_yn':'N', 'before_buy_yn':'N'}, True),
#            ({'cust_name':'WARD',  'card_yn':'Y', 'review_yn':'Y', 'before_buy_yn':'Y'}, True)]

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

import numpy.linalg as lin
lin.inv()