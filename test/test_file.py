import collections
import numpy as np
from _collections import defaultdict
from functools import partial
import csv
import time

class Tree:
    def __init__(self):
        self.inputs = inputs_csv()
        self.labels = self.inputs[0][0].keys()

    # labels의 True / False 개수를 구해주는 함수
    def classProbabilities(self, labels):
        total = len(labels)
        print(collections.Counter(labels)) # collection.Counter = key -> T/F , value = T/F의 개수
        return [cal / total for cal in collections.Counter(labels).values()]

    # 엔트로피를 계산하는 함수
    def entropy(self, labels):
        probability_YN = self.classProbabilities(labels)
        print(labels) # labels = 각 label 마다 list 로 구성된 T/F 값들
        print(probability_YN) # probability_YN = labels 마다의 T/F를 숫자형 데이터로 반환(binary 식)
        return sum(-p * np.log2(p) for p in probability_YN if p)  # 엔트로피 계산값 출력

    # target 된 label의 엔트로피 구하는 함수
    def target_data(self, inputs):
        groups = []
        print(inputs) # ({'cust_name': 'SCOTT', 'card_yn': 'Y', 'review_yn': 'Y', 'before_buy_yn': 'Y'}, True) ...
        for input in inputs:
            groups.append(input[-1])
        print(groups) # groups = list 로 구성된 True / False 값.
        return self.entropy(groups)

    # 파티션된(predictor) 엔트로피를 계산하는 함수
    def labels_entropy(self, predictors):
        total_cnt = sum(len(p) for p in predictors)
        return sum(self.entropy(p) * len(p) / total_cnt for p in predictors)

    # 트리나무 만들기
    def buildingtree(self, inputs, node=None):
        if node is None:
            node = self.labels
            self.nodes = Nodes()

        Tcnt = len([label for _ , label in inputs if label])
        Fcnt = len([label for _ , label in inputs if not label])
        if Tcnt == 0: return False
        if Fcnt == 0: return True
        if not node: return True if Tcnt >= Fcnt else False

        best_label = min(node, key=partial(self.nodes.predictors_data,inputs)) # best_label main version
        best_predictor = self.nodes.find_predictors(inputs, best_label)
        next_candidates = [ n for n in node if n != best_label]

        subtrees = { bestlabel : self.buildingtree(predictors,next_candidates)
                     for bestlabel, predictors in best_predictor.items()}
        print(subtrees)
        return {best_label:subtrees}

class Nodes(Tree):
    def __init__(self):
        super().__init__()

    # 파티션된(predictor) 노드의 엔트로피를 구하기 위한 함수
    def predictors_data(self, inputs, labels):
        groups = defaultdict(list)
        for input in inputs:
            key = input[0][labels]
            groups[key].append(input[-1])
        print(groups) # label 마다 { Y : [T,T,T,T] , N : [F] } 식으로 나타내줌
        return self.labels_entropy(groups.values())

    # best_predictor를 찾기 위한 함수
    def find_predictors(self, inputs, labels):
        groups = defaultdict(list)
        for input in inputs:
            key = input[0][labels]
            groups[key].append(input)
        return groups

    # best_predictor 추출하는 함수
    def select_predictors(self,inputs):
        global best_label
        best_infogain = 0.0
        for label in self.labels:
            pre_entro = self.target_data(inputs)
            rear_entro = self.predictors_data(inputs,label)
            infogain = pre_entro - rear_entro
            if infogain > best_infogain:
                best_infogain = infogain
                best_label = label
        return best_label

input_list = []
tmp_list = []

import codecs

def inputs_csv():  # 예진이껄로 변경하기
    with codecs.open('pretest.csv', "r", encoding='utf-8') as f:
        for tmp in csv.reader(f):
            tmp_list.append(tmp)
    labels = tmp_list[0]
    with codecs.open('pretest.csv', "r", encoding='utf-8') as f:
        for input in csv.DictReader(f):
            *predictors, target = labels
            input_list.append(({p:input[p] for p in predictors}, False if input[target] == 'No' else True))
    return input_list

start = time.time()
entropy = Nodes()
result_tree = entropy.buildingtree(entropy.inputs)
print(result_tree)  # 결정트리 출력
end = time.time()
print('Tree 생성 시간 : {} 초'.format(round(end-start,4)))