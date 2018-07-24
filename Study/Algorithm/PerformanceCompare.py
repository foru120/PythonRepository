import csv
import time

# csv file 로 부터 데이터를 가져오는 함수
def returnCsvData(filename):
    file = open('D:\\KYH\\02.PYTHON\\data\\' + filename, 'r', encoding='utf-8')
    return csv.reader(file)

# csv data 를 list 로 변환하는 함수
def csvDataToList(csvData):
    tempList = []
    for data in csvData:
        tempList.append(data)

    return tempList

# csv data 를 dict 로 변환하는 함수
def csvDataToDict(csvData, keyIndex):
    tempDict = {}
    for data in csvData:
        tempDict[data[keyIndex]] = data
    return tempDict


sTime = time.time()
###################################################################################################
## Data Initialization(List)
###################################################################################################
salesData = returnCsvData('sales.csv')
productsData = returnCsvData('products.csv')
timesData = returnCsvData('times.csv')

salesList = csvDataToList(salesData)  # 918843 건
productsList = csvDataToList(productsData)  # 72 건
timesList = csvDataToList(timesData)  # 1826 건
###################################################################################################
eTime = time.time()
print('init-time : %.02f' %(eTime-sTime))


sTime = time.time()
###################################################################################################
## Data Comparison(NL Join, ==)
###################################################################################################
cnt = 0
result1 = []
result2 = []
for tData in timesList:
    for sData in salesList:
        if tData[0] == sData[2]:
            result1.append([sData[0], sData[2], sData[6], tData[30]])
            cnt += 1

for rData in result1:
    for pData in productsList:
        if rData[0] == pData[0]:
            result2.append([rData[0], pData[1], rData[1], rData[2], rData[3]])
            cnt += 1
print('결과 집합 개수 : '+str(len(result2)), '수행 횟수 : '+str(cnt))
###################################################################################################
eTime = time.time()
print('comp-time(NL Join, ==) : %.02f' %(eTime-sTime))





sTime = time.time()
###################################################################################################
## Data Initialization(Dict)
###################################################################################################
salesData = returnCsvData('sales.csv')
productsData = returnCsvData('products.csv')
timesData = returnCsvData('times.csv')

salesList = csvDataToList(salesData)  # 918843 건
productsDict = csvDataToDict(productsData, 0)  # 72 건
timesDict = csvDataToDict(timesData, 0)  # 1826 건
###################################################################################################
eTime = time.time()
print('init-time : %.02f' %(eTime-sTime))


sTime = time.time()
###################################################################################################
## Data Comparison(Hash Join, in)
###################################################################################################
cnt = 0
result1 = []
result2 = []
for sData in salesList:
    if sData[2] in timesDict:
        result1.append([sData[0], sData[2], sData[6], timesDict.get(sData[2])[30]])
        cnt += 1

for rData in result1:
    if rData[0] in productsDict:
        result2.append([rData[0], productsDict.get(rData[0])[1], rData[1], rData[2], rData[3]])
        cnt += 1
print('결과 집합 개수 : '+str(len(result2)), '수행 횟수 : '+str(cnt))
###################################################################################################
eTime = time.time()
print('comp-time(Hash Join, in) : %.02f' %(eTime-sTime))