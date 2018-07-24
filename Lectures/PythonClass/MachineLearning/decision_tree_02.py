# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:12:18 2017
@author: stu
"""
# http://clearpal7.blogspot.kr/2016/07/3_10.html
# http://yujuwon.tistory.com/entry/%EC%9D%98%EC%82%AC-%EA%B2%B0%EC%A0%95-%ED%8A%B8%EB%A6%AC

import operator
from math import log


def createDataSet():
    dataSet = [['M', '30', 'NO', 'YES', 'NO', 'NO'],
               ['F', '20', 'YES', 'YES', 'YES', 'NO'],
               ['F', '20', 'YES', 'YES', 'NO', 'NO'],
               ['F', '40', 'NO', 'NO', 'NO', 'NO'],
               ['F', '30', 'NO', 'YES', 'NO', 'NO'],
               ['F', '30', 'NO', 'NO', 'YES', 'NO'],
               ['F', '20', 'NO', 'YES', 'NO', 'NO'],
               ['F', '20', 'NO', 'YES', 'YES', 'YES'],
               ['F', '30', 'YES', 'YES', 'NO', 'YES'],
               ['M', '40', 'YES', 'NO', 'YES', 'NO'],
               ['M', '20', 'NO', 'NO', 'NO', 'NO'],
               ['M', '30', 'NO', 'YES', 'YES', 'NO'],
               ['M', '20', 'YES', 'NO', 'NO', 'NO'],
               ['F', '30', 'YES', 'YES', 'NO', 'YES'],
               ['M', '30', 'YES', 'YES', 'YES', 'YES'],
               ['F', '30', 'YES', 'NO', 'NO', 'NO'],
               ['F', '30', 'NO', 'YES', 'YES', 'YES'],
               ['M', '20', 'YES', 'YES', 'NO', 'NO'],
               ['M', '40', 'YES', 'NO', 'YES', 'NO'],
               ['M', '30', 'NO', 'NO', 'NO', 'NO'],
               ['F', '30', 'YES', 'YES', 'NO', 'YES'],
               ['M', '30', 'YES', 'NO', 'YES', 'NO'],
               ['F', '40', 'NO', 'YES', 'YES', 'YES'],
               ['M', '30', 'NO', 'YES', 'NO', 'NO'],
               ['F', '30', 'YES', 'YES', 'YES', 'YES'],
               ['F', '40', 'YES', 'NO', 'YES', 'NO'],
               ['M', '40', 'YES', 'YES', 'NO', 'YES'],
               ['F', '40', 'YES', 'YES', 'NO', 'YES']]

    labels = ['GENDER', 'AGE', 'JOB_YN', 'MARRY_YN', 'CAR_YN', 'COUPON_YN']
    return dataSet, labels


def calcShannonEnt(dataSet):  # 분할 전 엔트로피 구하기
    numEntries = len(dataSet)  # row의 개수 28
    labelCounts = {}
    for feaVec in dataSet:
        currentLabel = feaVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # COUPON_YN labelCounts={'NO': 18, 'YES': 10}
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 분할 전 엔트로피:0.9402859586706309
    return shannonEnt


def splitDataSet(dataSet, axis, value):  # 변수의 각 속성에 따라 데이터 나누기(그 변수의 값은 제외하고)
    retDataSet = []  # GENDER F  [['20', 'YES', 'YES', 'YES', 'NO'],..,['40', 'YES', 'YES', 'NO', 'YES']
    for featVec in dataSet:  # GENDER M  [['30', 'NO', 'YES', 'NO', 'NO'],..,['40', 'YES', 'YES', 'NO', 'YES']]
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):  # 정보이득이 가장 큰 변수 출력
    numFeatures = len(dataSet[0]) - 1  # 한 row가 갖는 속성의 개수(COUPON_YN 제외) 5개
    baseEntropy = calcShannonEnt(dataSet)  # 분할 전 엔트로피
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # dataSet의 COUPON_YN변수 제외 속성 리스트
        # featList - ['M', 'F', 'F',.., 'F', 'M', 'F'] ['30', '20', '20',.., '40', '40', '40']
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # set() : 리스트 요소 중복제거 {'M', 'F'} {'20','30','40'} {'YES','NO'}
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 분할 후 엔트로피
        infoGain = baseEntropy - newEntropy  # 정보이득
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 가장 큰 정보이득을 갖는 변수의 위치


def majorityCnt(classList):  # 각 변수의 요소 중 개수가 가장 많은 것을 리턴
    classCount = {}
    for vote in classList:  # AGE ['30', '20', '20',.., '40', '40', '40']
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1  # COUPON_YN classCount {20':7,'30':13,'40':8}
    # [('20', 7), ('30', 13), ('40', 8)]
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 내림차순으로 정렬, 숫자가 가장 큰 '30'을 리턴


def createTree(dataSet, labels):
    # COUPON_YN 데이터 ['NO', 'NO', 'NO',.., 'NO', 'YES', 'YES']
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 종료조건1 : 모두 'yes'이거나 모두 'no'이면 종료하고 그 항목 리턴
        return classList[0]
    if len(dataSet[0]) == 1:  # 종료조건2 : [1,0,'yes']->[1,0]->[1] 데이터셋의 변수가 1개가 됐을 때 majorityCnt리턴
        return majorityCnt(classList)  # 둘 다 만족하지 못하면 정보이득이 가장 큰 조건으로
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 정보이득이 가장 큰 속성의 위치(ex.0~4)
    bestFeatLabel = labels[bestFeat]  # 정보이득이 가장 큰 속성(ex.'GENDER','AGE','JOB_YN','MARRY_YN','CAR_YN')
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 정보이득이 가장 큰 속성 이름 삭제
    featValues = [example[bestFeat] for example in dataSet]  # 정보이득이 가장 큰 속성 값들의 리스트
    uniqueVals = set(featValues)  # 그 속성의 중복값 제거 ex.{'M', 'F'} {'20','30','40'} {'YES','NO'}
    for value in uniqueVals:
        subLabels = labels[:]  # subLabels : 그 속성을 제외한 label 리스트
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                               bestFeat, value), subLabels)
        # ex. myTree={'MARRY_YN':{'YES':'aaa'}}
        # myTree['MARRY_YN']['NO']='bbb' {'MARRY_YN': {'YES': 'aaa', 'NO': 'bbb'}}
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]  # myTree의 맨 처음 key : MARRY_YN
    secondDict = inputTree[firstStr]  # MARRY_YN의 값 : {'YES': {'AGE':...}, 'NO': 'NO'}
    featIndex = featLabels.index(firstStr)  # 속성 리스트에서 MARRY_YN의 위치 3
    for key in secondDict.keys():  # MARRY_YN의 값의 key : ['YES','NO']
        if testVec[featIndex] == key:  # MARRY_YN에 대한 대답과 key가 같다면
            if type(secondDict[key]).__name__ == 'dict':  # 그 key의 값이 딕셔너리 타입이면 classify 재실행
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]  # 아니면, classLabel은 그 key의 값을 리턴
    return classLabel


def main():
    data, label = createDataSet()
    myTree = createTree(data, label)
    featlabel = ['GENDER', 'AGE', 'JOB_YN', 'MARRY_YN', 'CAR_YN']
    a = []
    for i in featlabel:
        a.append(input('{0}?'.format(i)).upper())
    answer = classify(myTree, featlabel, a)
    print(myTree)
    print('\n' + '쿠폰 반응여부 : ' + answer)


if __name__ == '__main__':
    main()