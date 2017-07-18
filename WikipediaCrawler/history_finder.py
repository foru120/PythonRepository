from bs4 import BeautifulSoup
from urllib.request import urlopen, urljoin, quote, unquote
from urllib.error import HTTPError
import re
import json
import threading
import os
import folium
from folium.plugins import HeatMap

from WikipediaCrawler.database import Database

###################################################################################################
## History 관련 클래스
###################################################################################################
class History(object):

    def __init__(self, base_url, keyword):
        self.base_url = base_url
        self.keyword = keyword
        self.db = Database(self.base_url, self.keyword)

    #  해당 문서의 History 내역을 search 하는 함수
    def getHistory(self):
        # request_url = u''.join([base_url, '/w/index.php?title='+keyword+'&limit=500&action=history'])
        data = self.db.getUrlName(self.db.getMaxUrlID())

        data_cnt = len(data)
        thread_list = []
        index = 0

        while True:
            cnt = 0
            # history 내역을 search 하는 thread 생성
            for i in range(0, data_cnt if data_cnt <= 3 else 3):
                thread_list.append(threading.Thread(target=self.historyThread, args=(data[index][0], data[index][1])))
                index += 1
                cnt += 1

            # thread 수행
            for thread in thread_list:
                thread.start()

            # 전체 thread 수행 완료까지 대기
            for thread in thread_list:
                thread.join()

            thread_list.clear()
            data_cnt -= cnt

            if data_cnt == 0:
                break

    #  개별 쓰레드들이 history 내역 크롤링하는 함수
    def historyThread(self, urlId, history_url):
        url = self.getURL(re.match('.*(\/wiki\/)(.*)', history_url).group(2))
        if url is not None:
            request_url = urljoin(self.base_url, url+'&limit=500')
        else:
            print('== histroy : URL is None!!!')
            return
        print('== history : ' + unquote(request_url))

        try:
            html = urlopen(request_url)
        except HTTPError as e:
            print(e, ' - ' + unquote(request_url))
            return

        bsObj = BeautifulSoup(html.read(), 'html.parser').find('ul', id='pagehistory')

        history_list = []

        for history in bsObj.find_all('li'):
            historyTime = history.find('a', class_='mw-changeslist-date')
            userLink = history.find('a', class_='mw-anonuserlink')
            historySize = history.find('span', class_='history-size')
            plusMinus =  history.find(['span', 'strong'], class_=re.compile('mw-plusminus-(pos|neg|null)'))

            if historyTime is not None:
                historyTime = historyTime.text

                if userLink is None:
                    userLink = history.find('a', class_='mw-userlink').text
                    countryData = {'country_name': '', 'city': '', 'latitude': '', 'longitude': ''}
                    isMember = 1
                else:
                    userLink = userLink.text
                    countryData = self.getCountry(userLink)
                    isMember = 0

                if historySize is not None:
                    if re.match('\((.*)\s(.*)\)', historySize.text) is not None:
                        historySize = re.match('\((.*)\s(.*)\)', historySize.text).group(1)
                    else:
                        historySize = ''
                else:
                    historySize = ''

                if plusMinus is not None:
                    if re.match('\((.*)\)', plusMinus.text) is not None:
                        plusMinus = re.match('\((.*)\)', plusMinus.text).group(1)
                    else:
                        plusMinus = ''
                else:
                    plusMinus = ''

                history_list.append({'urlId': urlId, 'historyTime': historyTime, 'userLink': userLink, 'countryName': countryData['country_name'],
                                     'city': countryData['city'], 'latitude': countryData['latitude'], 'longitude': countryData['longitude'],
                                     'isMember': isMember, 'historySize': historySize, 'plusMinus': plusMinus})

        self.db.insertHistory(history_list)

    # History URL search 하는 함수(리다이렉션 URL 처리 관련)
    def getURL(self, reform_url):
        request_url = quote(urljoin(self.base_url, '/wiki/'+reform_url), '/:?&=_')

        try:
            html = urlopen(request_url)
        except HTTPError as e:
            print(e, ' - ' + request_url)
            return

        bsObj = BeautifulSoup(html.read(), 'html.parser')

        return bsObj.find('li', {'id': 'ca-history'}).find('a').attrs['href']

    # 비 사용자 IP 주소를 통한 위치 리턴하는 함수(freegeoip API 사용)
    def getCountry(self, ipAddress):
        try:
            response = urlopen('http://freegeoip.net/json/'+ipAddress).read().decode('utf-8')
        except HTTPError as e:
            print(e, ' - ' + ipAddress)

        jsonObj = json.loads(response)

        return {'country_name': jsonObj.get('country_name'), 'city': jsonObj.get('city'), 'latitude': jsonObj.get('latitude'), 'longitude': jsonObj.get('longitude')}

    # 히트맵 출력 함수
    def printHeatmap(self):
        data = self.db.getLatLong()
        m = folium.Map([48., 5.], tiles='stamentoner', zoom_start=6)
        HeatMap(data).add_to(m)
        m.save(os.path.join('.', 'Heatmap.html'))