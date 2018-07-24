# ▣ html 의 구조
#  - hyper text markup language 의 약자이고 여러개의 태그를 연결해서 모아놓은 문서

print('')
print('====================================================================================================')
print('== 문제 260. a.html 의 내용을 수정하시오.')
print('====================================================================================================')
# <html><head><title>김남훈 대통령의 오늘 일정</title></head>
# <body>
# <p class="title"><b>The Dormouse's story</b></p>
# <p class="story">Once upon a time there were three little sisters; and their names were
# <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
# <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
# <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
# and they lived at the bottom of a well.</p>
# <p class="story">...</p>
# </body>


print('')
print('====================================================================================================')
print('== 문제 261. 태그 정리표를 보고 글씨를 진하게 출력하시오! (b 태그)')
print('====================================================================================================')
# <html><head><title>김남훈 대통령의 오늘 일정</title></head>
# <body>
# <p class="title"><b>The Dormouse's story</b></p>
# <p class="story">Once upon a time there were three little sisters; and their names were
# <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
# <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
# <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
# and they lived at the bottom of a well.</p>
# <p class="story">...</p>
# </body>


print('')
print('====================================================================================================')
print('== 문제 268. beautiple soup 모듈을 이용해서 위의 html 문서의 title 을 검색하시오.')
print('====================================================================================================')
from bs4 import BeautifulSoup

with open('PythonClass/html/P267_html.html', encoding='utf-8') as ht:
    soup = BeautifulSoup(ht, 'html.parser')

print(soup.title)


print('')
print('====================================================================================================')
print('== 문제 269. (점심시간 문제) title 의 text 만 검색되게 하시오.')
print('====================================================================================================')
from bs4 import BeautifulSoup

with open('PythonClass/html/P267_html.html', encoding='utf-8') as ht:
    soup = BeautifulSoup(ht, 'html.parser')

print(soup.title.text)


print('')
print('====================================================================================================')
print('== 문제 270. p태그에 대한 html 을 검색을하시오.')
print('====================================================================================================')
from bs4 import BeautifulSoup

with open('PythonClass/html/P267_html.html', encoding='utf-8') as ht:
    soup = BeautifulSoup(ht, 'html.parser')

print(soup.p)
print(soup.find('p'))


print('')
print('====================================================================================================')
print('== 문제 271. a태그에 대한 모든 데이터를 검색을하시오.')
print('====================================================================================================')
from bs4 import BeautifulSoup

with open('PythonClass/html/P267_html.html', encoding='utf-8') as ht:
    soup = BeautifulSoup(ht, 'html.parser')

print(soup.find_all('a'))


print('')
print('====================================================================================================')
print('== 문제 272. a태그에 href 링크의 url 만 긁어오시오.')
print('====================================================================================================')
from bs4 import BeautifulSoup

with open('PythonClass/html/P267_html.html', encoding='utf-8') as ht:
    soup = BeautifulSoup(ht, 'html.parser')

for tag in soup.find_all('a'):
    print(tag['href'])


print('')
print('====================================================================================================')
print('== 문제 273. text 만 긁어오시오.')
print('====================================================================================================')
from bs4 import BeautifulSoup

with open('PythonClass/html/P267_html.html', encoding='utf-8') as ht:
    soup = BeautifulSoup(ht, 'html.parser')

print(soup.text)


print('')
print('====================================================================================================')
print('== 문제 274. 위의 텍스트를 한줄로 나오게 하시오.')
print('====================================================================================================')
from bs4 import BeautifulSoup

with open('PythonClass/html/P267_html.html', encoding='utf-8') as ht:
    soup = BeautifulSoup(ht, 'html.parser')

print(soup.get_text(strip=True))


print('====================================================================================================')
print('== 문제 275. ecologicalpryamid.html문서 다운받아, D드라이브 밑에 넣어요')  #난 패키지폴더
print('====================================================================================================')


print('====================================================================================================')
print('== 문제 276. 위 html파일을 트리형으로 그림그려보기')
print('====================================================================================================')
#url : https://software.hixie.ch/utilities/js/live-dom-viewer/


print('====================================================================================================')
print('== 문제 277. 위 문서에서, text 1000000을 출력해보세요')
print('====================================================================================================')
with open("D:\\python\\source\\PythonClass\\Chap_15_web_scrolling\\ecologicalpyramid.html") as eco:
    soup = BeautifulSoup(eco,'html.parser')
# print(soup.find(class_="클래스이름"))
result = soup.find(class_="number")
print(result.get_text())    #1000000


print('====================================================================================================')
print('== 문제 278. 위 문서에서, number클래스에 있느 모든 텍스트 다 가져오시오.')
print('====================================================================================================')
result_all = soup.find_all(class_="number")

num_list = list()
for i in result_all:
    print(i)
    num_list.append(i.get_text())
print(num_list)


print('====================================================================================================')
print('== 문제 279. 위 결과에서 아래의 [1000]만 출력하라..')
print('====================================================================================================')
result = soup.find_all(class_="number")[2]
print(result.get_text())


print('====================================================================================================')
print('== 문제 280. ecological...html에서 fox 출력하라..')
print('====================================================================================================')
#name클래스의 4번째
result = soup.find_all(class_="name")[4]
print(result.get_text())

#트리형식으로 찾아가보기
result = soup.find_all("ul")[2]
print(result.li.div.text)


print('====================================================================================================')
print('== 문제 281. ecological...html안에 fox라는 텍스트가 있는지만 검색하시오.')
print('====================================================================================================')
result = soup.find(text="fox")
print(result)


print('====================================================================================================')
print('== 문제 284. deer 아래의 1000 이 출력되게 하시오.')
print('====================================================================================================')
from bs4 import BeautifulSoup

with open('PythonClass/html/ecologicalpyramid.html', encoding='utf-8') as ht:
    soup = BeautifulSoup(ht, 'html.parser')

print(soup.find_all('div', {'class': 'number'})[2].text)


print('====================================================================================================')
print('== 문제 285. ebs 레이디버그 시청자 게시판 페이지에서 dd 태그의 type2 클래스에 있는 href 링크 주소들을 전부 스크롤링하시오!')
print('====================================================================================================')
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

def fetch_list_url():
    list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?hmpMnuId=106"
    res = urlopen(Request(list_url)).read().decode("utf-8")
    soup = BeautifulSoup(res, "html.parser")
    for tag in soup.find_all('dd', class_='type2'):
        print(tag.find('a')['href'])

fetch_list_url()


print('====================================================================================================')
print('== 문제 286. 현재 페이지의 게시판 13개의 게시글이 전부 출력되게 하시오!')
print('====================================================================================================')
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

def fetch_list_url():
    list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?hmpMnuId=106"
    res = urlopen(Request(list_url)).read().decode("utf-8")
    soup = BeautifulSoup(res, "html.parser")
    for tag in soup.find('div', class_='postList').find_all('li', class_='spot_'):
        print(tag.find('p', class_='con').get_text(strip=True))

fetch_list_url()


print('====================================================================================================')
print('== 문제 287. 게시글 뿐만 아니라 게시날짜 정보도 같이 출력하시오!')
print('====================================================================================================')
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

def fetch_list_url():
    list_url = "http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?hmpMnuId=106"
    res = urlopen(Request(list_url)).read().decode("utf-8")
    soup = BeautifulSoup(res, "html.parser")
    for tag in soup.find('div', class_='postList').find_all('li', class_='spot_'):
        print(tag.find('p', class_='info').find('span', class_='date').get_text(strip=True), tag.find('p', class_='con').get_text(strip=True))

fetch_list_url()


print('====================================================================================================')
print('== 문제 288. 문제 287 코드에 for loop 를 추가해서 EBS 레이디 버그 게시판 글들을 모두 스크롤링 하시오.')
print('====================================================================================================')
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

def fetch_list_url():
    for i in range(1, 16):
        list_url = 'http://home.ebs.co.kr/ladybug/board/6/10059819/oneBoardList?c.page='+str(i)+'&hmpMnuId=106&searchKeywordValue=0&bbsId=10059819&searchKeyword=&searchCondition=&searchConditionValue=0&'
        res = urlopen(Request(list_url)).read().decode("utf-8")
        soup = BeautifulSoup(res, "html.parser")
        for tag in soup.find('div', class_='postList').find_all('li', class_='spot_'):
            print(tag.find('p', class_='info').find('span', class_='date').get_text(strip=True), tag.find('p', class_='con').get_text(strip=True))

fetch_list_url()


print('====================================================================================================')
print('== 문제 299. (점심시간 문제) 상세기사 검색하는 url 을 아무거나 선택하고 이 url 을 가지고 상세기사 내용을 출력하게하는 '
      'fetch_list_url2() 라는 함수를 생성하시오.')
print('====================================================================================================')
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

def fetch_list_url():
    params = list()
    for i in range(0,19):
        url_format = "http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq={}".format(i)
        url = Request(url_format)  # url요청에 따른 http통신 헤더값을 얻기 위함
        res = urlopen(url).read().decode("utf-8")  # 영어가 아닌, 한글을 긁어오기 위해 디코딩
        soup = BeautifulSoup(res, 'html.parser')  # res html문서를 Bs모듈로

        for tag in soup.find('ul', class_='search-result-list').find_all('dt'):
            params.append(tag.find('a')['href'])
    return params

def fetch_list_url2():
    for url_name in fetch_list_url():
        print(url_name)
        url = Request(url_name)  # url요청에 따른 http통신 헤더값을 얻기 위함
        res = urlopen(url).read().decode("utf-8")  # 영어가 아닌, 한글을 긁어오기 위해 디코딩
        soup = BeautifulSoup(res, 'html.parser')  # res html문서를 Bs모듈로
        print(soup.find('div', class_='article-text').text)

fetch_list_url2()


print('====================================================================================================')
print('== 문제 300. fetch_list_url() 함수가 리턴하는 params 의 url 값들을 fetch_list_url2() 에서 호출해 오게 하시오.')
print('====================================================================================================')
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

def fetch_list_url():
    params = list()
    for i in range(0, 19):
        url_format = "http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq={}".format(
            i)
        url = Request(url_format)  # url요청에 따른 http통신 헤더값을 얻기 위함
        res = urlopen(url).read().decode("utf-8")  # 영어가 아닌, 한글을 긁어오기 위해 디코딩
        soup = BeautifulSoup(res, 'html.parser')  # res html문서를 Bs모듈로

        for tag in soup.find('ul', class_='search-result-list').find_all('dt'):
            if tag.find('a') is not None:
                params.append(tag.find('a')['href'])
    return params

def fetch_list_url2():
    for url_name in fetch_list_url():
        print(url_name)
        url = Request(url_name)  # url요청에 따른 http통신 헤더값을 얻기 위함
        res = urlopen(url).read().decode("utf-8")  # 영어가 아닌, 한글을 긁어오기 위해 디코딩
        soup = BeautifulSoup(res, 'html.parser')  # res html문서를 Bs모듈로
        print(soup.find('div', class_='article-text').text)

fetch_list_url2()


print('====================================================================================================')
print('== 문제 301. 원하는 키워드를 가지고 파싱하시오')
print('====================================================================================================')
from urllib.request import Request, urlopen, quote
from bs4 import BeautifulSoup

def fetch_list_url():
    params = list()
    keyword = '인공지능'
    for i in range(0,19):
        # url_format = "http://search.hani.co.kr/Search?command=query&keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq={}".format(i)
        url_format = quote("http://search.hani.co.kr/Search?command=query&keyword="+keyword+"&media=news&sort=d&period=all&datefrom=2000.01.01&dateto=2017.05.17&pageseq={}".format(i), '/:?&=_')
        url = Request(url_format)  # url요청에 따른 http통신 헤더값을 얻기 위함
        res = urlopen(url).read().decode("utf-8")  # 영어가 아닌, 한글을 긁어오기 위해 디코딩
        soup = BeautifulSoup(res, 'html.parser')  # res html문서를 Bs모듈로

        for tag in soup.find('ul', class_='search-result-list').find_all('dt'):
            if tag.find('a') is not None:
                params.append(tag.find('a')['href'])
    return params

def fetch_article():
    for url_name in fetch_list_url():
        print(url_name)
        url = Request(url_name)  # url요청에 따른 http통신 헤더값을 얻기 위함
        res = urlopen(url).read().decode("utf-8")  # 영어가 아닌, 한글을 긁어오기 위해 디코딩
        soup = BeautifulSoup(res, 'html.parser')  # res html문서를 Bs모듈로
        print(soup.find('div', class_='article-text').text)

def content_to_file(content):
    with open('D:\\KYH\\02.PYTHON\\data\\article.txt', 'a', encoding='utf-8') as file:
        file.write(content)

fetch_article()


print('====================================================================================================')
print('== 문제 302. 중앙일보에서 파싱하시오.')
print('====================================================================================================')
from urllib.request import Request, urlopen, quote
from bs4 import BeautifulSoup

def fetch_list_url():
    params = list()
    for i in range(1, 101):
        url_format = "http://search.joins.com/JoongangNews?page={}&Keyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5&SortType=New&SearchCategoryType=JoongangNews&MatchKeyword=%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5".format(i)
        url = Request(url_format)  # url요청에 따른 http통신 헤더값을 얻기 위함
        res = urlopen(url).read().decode("utf-8")  # 영어가 아닌, 한글을 긁어오기 위해 디코딩
        soup = BeautifulSoup(res, 'html.parser')  # res html문서를 Bs모듈로

        for tag in soup.find('ul', class_='list_default').find_all('strong', class_='headline mg'):
            if tag.find('a') is not None:
                params.append(tag.find('a')['href'])
    return params

def article_crawling():
    for url_name in fetch_list_url():
        print(url_name)
        url = Request(url_name)  # url요청에 따른 http통신 헤더값을 얻기 위함
        res = urlopen(url).read().decode("utf-8")  # 영어가 아닌, 한글을 긁어오기 위해 디코딩
        soup = BeautifulSoup(res, 'html.parser')  # res html문서를 Bs모듈로
        content_to_file(soup.find('div', id='article_body').text)

def content_to_file(content):
    with open('D:\\KYH\\02.PYTHON\\data\\article.txt', 'a', encoding='utf-8', newline='\n') as file:
        file.write(content)

article_crawling()


print('====================================================================================================')
print('== 문제 316. 답변을 answer 변수에 담으시오!')
print('====================================================================================================')
import urllib

def fetch_list_url():
    params = list()
    for i in range(1,30):   #1번부터 30번페이지 까지 순환하라
        url_format = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp"
        request_header = urllib.parse.urlencode({"page": i})  # 출력 >> page=1 .. 30
        request_header = request_header.encode("utf-8") #위에서 출력한 값 자신을 utf-8로 인코딩하겠다.   b'page=1'
        url = urllib.request.Request(url_format,request_header)  #url주소의 html과 페이지정보 두개 요청드립니다.   # 출력 >> <urllib.request.Request object at 0x00620A10>
        res = urllib.request.urlopen(url).read().decode("utf-8")
        soup = BeautifulSoup(res,'html.parser')
        li_find = soup.find_all('li',class_='pclist_list_tit2')
        for idx in range(len(li_find)):
            href_find = li_find[idx].find("a")["href"]
            href_address = re.search("[0-9]{14}",href_find).group()   #숫자 0~9까지 해당하는 값 / 14개를 href_find로부터 가져와라.
            params.append(href_address)
    return params

fetch_list_url()

def fetch_href_url():
    detail_url = 'https://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp'
    request_header = urllib.parse.urlencode({'RCEPT_NO': '20170504003012'})
    request_header = request_header.encode('utf-8')
    url = urllib.request.Request(detail_url, request_header)
    res = urllib.request.urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(res, 'html.parser')
    div_find = soup.find('div', class_='form_table')
    tables_find = div_find.find_all('table')
    td_find = tables_find[0].find_all('td')
    question_find = tables_find[1].find('div', class_='table_inner_desc')
    answer_find = tables_find[2].find('div', class_='table_inner_desc')

    get_date = td_find[1].get_text()
    get_title = td_find[0].get_text()
    get_question = question_find.get_text(strip=True)
    get_answer = answer_find.get_text(strip=True)

    print(get_date, get_title)
    print(get_question)
    print(get_answer)

fetch_href_url()


print('====================================================================================================')
print('== 문제 318. get_save_path() 함수를 사용해서 위의 결과를 텍스트 파일로 생성할 수 있도록 하시오.')
print('====================================================================================================')
import urllib.request  # 웹브라우저에서 html 문서를 얻어오기위해 통신하는 모듈
from  bs4 import BeautifulSoup  # html 문서 검색 모듈
import os
import re

def get_save_path():
    save_path = input('Enter the file name and file location : ')
    save_path = save_path.replace('\\', '/')
    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0])
    return save_path


def fetch_list_url():
    params = []
    for j in range(1, 30):

        list_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp"

        request_header = urllib.parse.urlencode({"page": j})
        # print (request_header) # 결과 page=1, page=2 ..

        request_header = request_header.encode("utf-8")
        # print (request_header) # b'page=29'

        url = urllib.request.Request(list_url, request_header)
        # print (url) # <urllib.request.Request object at 0x00000000021FA2E8>

        res = urllib.request.urlopen(url).read().decode("utf-8")

        soup = BeautifulSoup(res, "html.parser")
        soup2 = soup.find_all("li", class_="pclist_list_tit2")
        for soup3 in soup2:
            soup4 = soup3.find("a")["href"]
            params.append(re.search("[0-9]{14}", soup4).group())

    return params


def fetch_list_url2():
    params2 = fetch_list_url()

    f = open('D:\\KYH\\02.PYTHON\\crawled_data\\crawled_data.txt', 'w', encoding='utf-8')

    for i in params2:
        detail_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp"

        request_header = urllib.parse.urlencode({"RCEPT_NO": str(i)})
        request_header = request_header.encode("utf-8")

        url = urllib.request.Request(detail_url, request_header)
        res = urllib.request.urlopen(url).read().decode("utf-8")
        soup = BeautifulSoup(res, "html.parser")
        soup2 = soup.find("div", class_="form_table")

        tables = soup2.find_all("table")
        table0 = tables[0].find_all("td")
        table1 = tables[1].find("div", class_="table_inner_desc")
        table2 = tables[2].find("div", class_="table_inner_desc")

        date = table0[1].get_text()
        title = table0[0].get_text()
        question = table1.get_text(strip=True)
        answer = table2.get_text(strip=True)

        f.write('=='*30 + '\n')
        f.write(title + '\n')
        f.write(date + '\n')
        f.write(question + '\n')
        f.write(answer + '\n')
        f.write('=='*30 + '\n')
    f.close

fetch_list_url2()


print('====================================================================================================')
print('== 문제 319. 방금 생성한 텍스트 파일을 가지고 R에서 워드 클라우드로 그리시오!')
print('====================================================================================================')
# install.packages("KoNLP")
# install.packages("wordcloud")
# install.packages("plyr")
#
# library(KoNLP)
# library(wordcloud)
# library(plyr)
# getwd()
# ## 현재 작업위치
# ## setwd('script')
# ## 작업위치 지정(이걸 하면 디렉토리를 일부러 다쓸필요가 없고 그 위치에 있는 파일명만 입력하면 됩니다,
# seoul < - readLines('crawled_data.txt', encoding="UTF-8")
# ## 텍스트파일을 불러오는데, 한글이 깨지므로 한글형식으로 불러오는 작업입니다.
# data2 < - sapply(seoul, extractNoun, USE.NAMES = F)
# data3 < - unlist(data2)
# data3 < - Filter(function(x)
# {nchar(x) >= 2}, data3)
# data3 < - gsub("\\d+", "", data3)
# data3 < - gsub("\\(", "", data3)
# data3 < - gsub("\\)", "", data3)
# data3 < - gsub("[A-Za-z]", "", data3)
# write(unlist(data3), "seoul3.txt")
# data4 < - read.table("seoul3.txt")
# wordcount < - table(data4)
# palete < - brewer.pal(9, "Set1")
# wordcloud(names(wordcount), freq=wordcount, scale=c(5, 1), rot.per = 0.1, min.freq = 1, random.order = F, color = T, colors = palete)