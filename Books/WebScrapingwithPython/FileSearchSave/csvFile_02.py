from urllib.request import urlopen
from io import StringIO
import csv

data = urlopen('http://pythonscraping.com/files/MontyPythonAlbums.csv').read().decode('ascii', 'ignore')
dataFile = StringIO(data)
# csvReader = csv.reader(dataFile)
dictReader = csv.DictReader(dataFile)  # csv file 을 key:value 형태로 리턴

print(dictReader.fieldnames)  # DictReader 를 사용하면 각 필드이름은 fieldnames 에 저장

for row in dictReader:
    print(row)