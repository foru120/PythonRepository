from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import datetime
import random
import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', passwd='mysql', db='mysql', charset='utf8')
cur = conn.cursor()
cur.execute('use scraping')

random.seed(datetime.datetime.now())

def store(title, content):
    cur.execute('insert into pages(title, content) values(\"%s\", \"%s\")', (title, content))
    cur.connection.commit()

def getLinks(articleUrl):
    html = urlopen('http://en.wikipedia.org'+articleUrl)
    bsObj = BeautifulSoup(html.read(), 'html.parser')
    title = bsObj.find('h1').get_text()
    content = bsObj.find('div', {'id':'mw-content-text'}).find('p').text
    store(title, content)
    return bsObj.find('div', {'id':'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))

links = getLinks('/wiki/Kevin_Bacon')

try:
    while len(links) > 0:
        newArticle = links[random.randint(0, len(links)-1)].attrs['href']
        print(newArticle)
        links = getLinks(newArticle)
finally:
    cur.close()
    conn.close()