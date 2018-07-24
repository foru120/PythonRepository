import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://en.wikipedia.org/wiki/Comparison_of_text_editors')
bsObj = BeautifulSoup(html.read(), 'html.parser')

# 비교 테이블은 현재 페이지의 첫 번째 테이블입니다.
table = bsObj.find_all('table', {'class':'wikitable'})[0]
rows = table.find_all('tr')

with open('downloads/editors.csv', 'wt', encoding='utf-8') as csvFile:
    writer = csv.writer(csvFile)
    for row in rows:
        csvRow = []
        for cell in row.find_all(['td', 'th']):
            csvRow.append(cell.get_text())
        writer.writerow(csvRow)