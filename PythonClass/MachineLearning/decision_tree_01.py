# 노드는 의사결정트리를 분류 할 데이터가 있는 마디를 의미합니다. 모든 데이터가 들어있는 맨 처음 첫 노드를 부모 노드라고 하고
# 부모 노드로부터 분류되어 아래로 생성된 노드를 자식 노드 라고 합니다.
# 양 갈래로 노드를 분류를 아래쪽으로 계속 진행해서 완벽하게 분류 된 상태이거나 분류하는 것이 의미 없을 때 노드 생성을 종료하는 과정을 거치게 되는데
# 노드 구조가 마치 나무와 같다고 하여 의사결정트리(Decision Tree) 라고 합니다. (더 자세한 사항은 첨부한 PDF 파일을 참조하세요)
# Category 형 데이터 ( 범주형 데이터 : 성별, 질병발생여부와 같이 숫자로 되어 있지 않거나
# 숫자로 되어있더라도 특정한 명목을 나타내는 데이터를 의미 )를 분류할 땐
# 엔트로피와 Greedy 알고리즘을 기반으로 하는 c4.5 ( ID3에서 시작해서 c4.5를 거쳐 현재는 c5.0까지 개량 ) 알고리즘으로 트리를 제작합니다.
# 각 노드 단계별 데이터의 분류는 엔트로피를 계산한 정보획득량을 기반으로 분류하고 노드 생성은 greedy 알고리즘을 사용합니다.
# c4.5를 사용한 간단한 의사결정트리를 생성해서 수업시간에 실행했던 쿠폰사용여부 데이터를 분류해보고
# 데이터에 없는 값을 물어 봤을 때 쿠폰사용여부를 출력하도록 합니다.(예측)

# 의사결정트리 코드

import math               # 엔트로피 계산에 로그함수를 쓰기 위해 math 모듈을 불러온다.

import collections        # 분석 컬럼별 확률을 계산하기 위해 만드는 class_probabilities를 함수로 만들기 위해 사용하는 모듈을 불러온다.
                          # 자세히 설명하면 collection.Counter 함수를 사용하기 위함인데
                          # collection.Counter() 함수는 Hashing(책 p.110 참조) 할 수 있는 오브젝트들을
                          # count 하기 위해 서브 클래스들을 딕셔너리화 할 때 사용한다.

from functools import partial         # c5.0(엔트로피를 사용하는 알고리즘) 알고리즘 기반의 의사결정트리를
                                      # 제작하는 트리제작함수인 build_tree_id3에서 사용하는데
                                      # functools.partial() 함수는 기준 함수로 부터 정해진 기존 인자가 주어지는 여러개의 특정한 함수를 만들고자 할 때 사용한다.
                                      # http://betle.tistory.com/entry/PYTHON-decorator를 참조

from collections import defaultdict         # 팩토리 함수는 특정 형의 데이터 항목을 새로 만들기 위해 사용되는데
                                            # defaultdict(list) 함수는 missing value(결측값)을 처리 하기 위한
                                            # 팩토리 함수를 불러오는 서브 클래스를 딕셔너리화 할 떄 사용한다.


def entropy(class_probabilities):            # 해당 컬럼의 확률을 입력받아 해당 클래스의 엔트로피를 계산하는 함수
    return sum(-p * math.log(p, 2)           # 정보획득량을 계산하기 위한 조건부 엔트로피를 합산하는 수식
               for p in class_probabilities if p)        # 해당 컬럼에 빈도수가 존재하지 않으면 0인데 이경우 조건부 확률도 0이되게 되고
                                                         # 그럴 경우 log함수에서 0을 입력하면 에러가 발생한다.
                                                         # 따라서 0이 아닌 경우에만 엔트로피를 계산한다.

def class_probabilities(labels):       # 엔트로피 계산에 사용하는 컬럼의 확률을 계산하는 함수
    total_count = len(labels)          # 전체 count 수 = 해당 컬럼의 총 길이
                                       # 분석에 사용하는 데이터 구조가 다음과 같기 때문에 전체 count수는 len(labels)로 할 수 있다.
                                       # 데이터셋 = [ ( {데이터가 되는 컬럼의 키와 값으로 구성된 딕셔너리}, 분석타겟컬럼의 값  ) , ..... ]

    return [count / total_count for count in collections.Counter(labels).values()]      #계산한 확률을 list형으로 반환


def data_entropy(labeled_data):         # 전체 데이셋의 엔트로피
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)


def partition_entropy(subsets):         # 파티션된 노드들의 엔트로피
    total_count = sum(len(subset) for subset in subsets)        # subset은 라벨이 있는 데이터의 리스트의 리스트이다. 이것에 대한 엔트로피를 계산한다.
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

#########################   데이터셋   #################################
# 데이터셋 = [ ( {데이터가 되는 컬럼의 키와 값으로 구성된 딕셔너리}, 분석타겟컬럼의 값  ) , ..... ]
# 분석 타겟 컬럼 : 라벨(label)
# 분석에 사용하는 데이터가 되는 컬럼 : 어트리뷰트(attribute) - 속성
# 분석에 사용하는 데이터의 값 : inputs

inputs = []          # 최종적으로 사용할 데이터셋의 형태가 리스트여야 하기 때문에 빈 리스트를 생성합니다.

import csv
file=open("D:\\KYH\\02.PYTHON\\data\\fatliver.csv", "r")          # csv 파일로 데이터셋을 불러옴
fatliver=csv.reader(file)
inputss=[]
for i in fatliver:
    inputss.append(i)        # 데이터 값

labelss = ['age', 'gender', 'drink', 'smoke', 'Fatliver']        # 데이터의 라벨(컬럼명)

for data in inputss:        # 위처럼 리스트로 된 데이터값과 리스트로된 라벨(컬럼명)을 분석에 맞는 데이터형태로 바꾸는 과정.
    temp_dict = {}          # 데이터셋 = [ ( {데이터가 되는 컬럼의 키와 값으로 구성된 딕셔너리}, 분석타겟컬럼의 값  ) , ..... ] 의 형태로 되어있어야 분석할 수 있다.
    c=len(labelss)-1        # 데이터셋의 최종값을 타겟변수로 두었기 때문에 타겟변수는 데이터값 딕셔너리에 넣지 않습니다. 분석타겟변수의 위치를 잡아주는 값
    for i in range(c):      # 타겟변수를 제외한 나머지 변수들로 딕셔너리에 데이터를 입력
        if i != c:          # 생성한 딕셔너리와 넣지 않은 타겟변수를 분석을 위한 큰 튜플안에 입력
            temp_dict[labelss[i]] = data[i]
    inputs.append(tuple((temp_dict, True if data[c] == 'yes' else False)))          #


def partition_by(inputs, attribute):        # attribute에 따라 inputs(데이터)를 파티션하는 함수
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]           # 특정 attribute의 값을 불러오고 해당 attribute의 input값을 list에 추가한다.
        groups[key].append(input)
    return groups


def partition_entropy_by(inputs, attribute):        # 위에서 attribute에 따라 inputs를 파티션한 파티션의 엔트로피를 계산하는 함수
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())



def classify(tree, input):                        # 분류기 선언, tree의 input값을 분류
    if tree in [True, False]:                     # 잎 노드면 값을 반환
        return tree

    attribute, subtree_dict = tree                # 위 경우가 아니면 키로 attribute, input 으로 서브트리를 나타내는 딕셔너리로 파티션 분할 실행

    subtree_key = input.get(attribute)            # 입력된 데이터 변수 중 하나가 기존에 관찰되지 않은 변수면 None을 입력

    if subtree_key not in subtree_dict:           # 키에 해당하는 서브트리가 존재하지 않으면
        subtree = subtree_dict[subtree_key]       # None 서브트리를 사용
    subtree = subtree_dict[subtree_key]           # 서브트리를 선택
    return classify(subtree, input)               # 이 과정을 재귀를 통해 잎 노드가 반환될 때까지 계속 수행


def build_tree_id3(inputs, split_candidates=None):
    if split_candidates is None:                  # 파티션이 첫 단계면 입력된 데이터의 모든 변수를 파티션 기준 후보로 설정
        split_candidates = inputs[0][0].keys()

    num_inputs = len(inputs)                      # 입력된 데이터에서 True, False 개수를 체크
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues

    if num_trues == 0:
        return False                              # true가 없다면 false 잎 노드를 반환
    if num_falses == 0:
        return True                               # false가 없다면 true 잎 노드를 반환

    if not split_candidates:
        return num_trues >= num_falses            # 만약 사용할 변수가 없으면 많은 수를 반환

    best_attribute = min(split_candidates, key=partial(partition_entropy_by, inputs))     # 가장 적합한 변수(attribute)를 기준으로 파티션 시작
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]
    subtrees = {attribute_value: build_tree_id3(subset, new_candidates) for attribute_value, subset in      # 재귀적으로 돌아가면서 서브 트리를 구축
                partitions.items()}
    subtrees[None] = num_trues > num_falses  # 기본값
    return (best_attribute, subtrees)


tree = build_tree_id3(inputs)

def bot():
    print('------------------------------------------------------------')
    print('     지방간 여부 판단 봇입니다. 데이터를 수집하겠습니다.')
    print('------------------------------------------------------------')
    gender_input = input(' 당신의 성별은? (남자,여자) ')
    age_input = input(' 당신의 연령대는? (30대,40대,50대) ')
    smoke_input = input(' 당신의 흡연여부는? (금연,흡연) ')
    drink_input = input(' 당신의 음주여부는? (음주적음,음주많음) ')
    answer = classify(tree, {"gender" : gender_input, "age" : age_input, "drink" : drink_input, "smoke" : smoke_input})
    for x in range(10,0,-1):
        for y in range(x):
            print('★', end = "")
        print()
    print('60초 후에 공개합니다! 광고보고 오시죠!')
    for i in range(1, 10):
        for j in range(i):
            print('★', end="")
        print()
    if answer is True:
        print('당신은...')
        print('지방간이.....')
        print('"맞습니다"')
    elif answer is False:
        print('당신은')
        print('지방간이')
        print('"아닙니다"')

bot()