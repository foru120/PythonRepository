#-*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import operator
import time

class KMACrawler:
    FILE_PATH = 'D:\\KYH\\02.PYTHON\\crawled_data\\'

    def __init__(self):
        self.location_list = {}
        self.year_list = {}
        self.factor_list = {}
        self.crawling_list = {}
        self.data = {}
        self.default_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp'
        self.crawled_url = 'http://www.kma.go.kr/weather/climate/past_table.jsp?stn={}&yy={}&obs={}'

    # 크롤링 된 데이터를 파일로 저장하는 함수
    def data_to_file(self):
        with open(KMACrawler.FILE_PATH + "kma_crawled.txt", "a", encoding="utf-8") as file:
            file.write('======================================================\n')
            for key, value in self.data.items():
                file.write('>> ' + key[0] + ', ' + key[1] + ', ' + key[2] + '\n')
                for v in value:
                    file.write(','.join(v) + '\n')
            file.write('======================================================\n\n')

    # 지점, 연도, 요소에 데이터 가져오는 함수
    def get_kma_data(self):
        res = urlopen(Request(self.default_url)).read()
        soup = BeautifulSoup(res, 'html.parser')

        location = soup.find('select', id='observation_select1')
        year = soup.find('select', id='observation_select2')
        factor = soup.find('select', id='observation_select3')

        for tag in location.find_all('option'):
            if tag.text != '--------':
                self.location_list[tag['value']] = tag.text

        for tag in year.find_all('option'):
            self.year_list[tag['value']] = tag.text

        for tag in factor.find_all('option'):
            self.factor_list[tag['value']] = tag.text

        for loc_key, loc_value in self.location_list.items():
            for year_key, year_value in self.year_list.items():
                for fac_key, fac_value in self.factor_list.items():
                    self.crawling_list[(loc_key, year_key, fac_key)] = (loc_value, year_value, fac_value)

    # 크롤링 수행하는 메인 함수
    def play_crawling(self):
        print('크롤링을 위한 데이터를 수집 중입니다...')
        self.get_kma_data()
        print('크롤링을 위한 데이터 수집 완료 !!!')

        print('크롤링을 시작합니다...')
        for key, value in sorted(self.crawling_list.items(), key=operator.itemgetter(0)):
            res = urlopen(Request(self.crawled_url.format(*key))).read()
            soup = BeautifulSoup(res, 'html.parser')

            print('현재 키워드 : {}, {}, {}'.format(*value))
            for tr_tag in soup.find('table', class_='table_develop').find('tbody').find_all('tr'):
                if self.data.get(value) is None:
                    self.data[value] = []
                self.data[value].append(['' if td_tag.text == '\xa0' else td_tag.text for td_tag in tr_tag.find_all('td') if td_tag.has_attr('scope') is False])

            print('{}, {}, {} 에 대한 데이터 저장...'.format(*value))
            self.data_to_file()
            self.data.clear()
            print('저장 완료!!!\n\n')
            time.sleep(2)
        print('크롤링 완료 !!!')

crawler = KMACrawler()
crawler.play_crawling()