from bs4 import BeautifulSoup
import re
import pymysql
from urllib.request import urlopen

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='mysql', db='mysql', charset='utf-8')
cur = conn.cursor()
cur.execute('use wikipedia')

def pageScraped(url):
    cur.execute('select * from pages where url = %s', (url))

    if cur.rowcount == 0:
        return False

    page = cur.fetchone()

    cur.execute('select * from links where fromPageId = %s', (int(page[0])))
    if cur.rowcount == 0:
        return False
    return True

def insertPageIfNotExists(url):
    cur.execute('select * from pages where url = %s', (url))
    if cur.rowcount == 0:
        cur.execute('insert into pages(url) values(%s)', (url))
        conn.commit()
        return cur.lastrowid
    else:
        return cur.fetchone()[0]

def insertLink(fromPageId, toPageId):
    cur.execute('select * from links where fromPageId = %s and toPageId = %s',
                (int(fromPageId), int(toPageId)))
    if cur.rowcount == 0:
        cur.execute('insert into links (fromPageId, toPageId) values (%s, %s)',
                    (int(fromPageId), int(toPageId)))
        conn.commit()

def getLinks(pageUrl, recursionLevel):
    global pages
    if recursionLevel > 4:
        return
    pageId = insertPageIfNotExists(pageUrl)
    html = urlopen('http://en.wikipedia.org'+pageUrl)
    bsObj = BeautifulSoup(html.read(), 'html.parser')

    for link in bsObj.find_all('a', href=re.compile('^(/wiki/)((?!:).)*$')):
        insertLink(pageId, insertPageIfNotExists(link.attrs['href']))
        if not pageScraped(link.attrs['href']):
            # 새 페이지를 만났으니 추가하고 링크를 검색합니다
            newPage = link.attrs['href']
            print(newPage)
            getLinks(newPage, recursionLevel+1)