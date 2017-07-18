from urllib.request import quote, unquote, urlopen, urljoin
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import threading

from WikipediaCrawler.file_io import *

###################################################################################################
## Link 관련 클래스
###################################################################################################
class Link(object):

    def __init__(self, base_url, keyword):
        self.base_url = base_url
        self.keyword = keyword
        self.db = Database(self.base_url, self.keyword)

    # 해당 페이지의 링크를 search 하는 함수
    def getLinks(self, urlName, recursionLevel):
        if recursionLevel >= 2:
            return

        unquote_url = unquote(urljoin(self.base_url, urlName))
        if len(unquote_url) > 255:  # url 길이가 255 가 넘지 않으면
            return
        urlId = self.db.insertUrlIfNotExists(unquote_url)

        print(unquote(urljoin(self.base_url, urlName)))
        request_url = quote(urljoin(self.base_url, urlName), '/:?&=_')  # url 타입 문자열 인코딩 -> encoding 인수 default : utf-8

        try:
            html = urlopen(request_url)
        except HTTPError as e:
            print(e, ' - ' + request_url)
            return

        bsObj = BeautifulSoup(html.read(), 'html.parser').find('div', {'id': 'bodyContent'})

        link_list = []

        if recursionLevel == 0:
            link_list.append(unquote(request_url))

        for tag in bsObj.find_all(['p', 'li']):
            for link in tag.find_all('a', href=re.compile('^(/wiki/)((?!:).)*$')):
                link = unquote(urljoin(self.base_url, link.attrs['href']))  # url 타입 문자열 디코딩 -> encoding 인수 default : utf-8

                if not self.db.urlScraped(link):
                    # 새 페이지를 만났으니 추가하고 링크를 검색합니다.
                    print('== link : ' + link)
                    if len(link) <= 255:
                        link_list.append(link)
                        self.db.insertLink(urlId, self.db.insertUrlIfNotExists(link))
                        self.getLinks(link, recursionLevel+1)
                    else:
                        print('== 비 정상적인 URL 입니다.' + link)
                else:
                    self.db.insertLink(urlId, self.db.insertUrlIfNotExists(link))
                    print('Skipping : ' + link + ' found on ' + urlName)

        list_thread = threading.Thread(target=self.linkUrlToFile, args=(link_list))
        list_thread.start()

    # 찾은 링크의 내용을 파일로 저장하는 함수
    def linkUrlToFile(self, link_list):
        link_cnt = len(link_list)
        index = 0
        thread_list = []

        while True:
            cnt = 0
            # 쓰레드 생성
            for i in range(0, link_cnt if link_cnt <= 10 else 10):
                thread_list.append(threading.Thread(target=self.linkThread, args=(link_list[index])))
                index += 1
                cnt += 1

            # 쓰레드 수행
            for thread in thread_list:
                thread.start()

            # 쓰레드 대기
            for thread in thread_list:
                thread.join()

            thread_list.clear()
            link_cnt -= cnt

            if link_cnt == 0:
                break

    def linkThread(self, link):
        print('== file : ' + link)
        html = urlopen(quote(urljoin(self.base_url, link), '/:?&=_'))
        bsObj = BeautifulSoup(html.read(), 'html.parser').find('div', id='mw-content-text')

        cnt = 1
        for p in bsObj.find_all('p'):
            if cnt == 1:
                File.write_file('ScrapedFiles/' + self.keyword + '#' + re.match('.*(\/wiki\/)(.*)', link).group(2) + '.txt', p.text)
            else:
                File.append_to_file('ScrapedFiles/' + self.keyword + '#' + re.match('.*(\/wiki\/)(.*)', link).group(2) + '.txt', p.text)
            cnt += 1

        if cnt == 1:
            print('*** 저장할 데이터가 없습니다 ***')