from bs4 import BeautifulSoup
from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
import time                     # 중간중간 sleep 을 걸어야 해서 time 모듈 import
import calendar
import operator
import random

class Ismart_crawler:
    ID = '0422202556'
    PW = 'bb772771'
    FILE_PATH = 'D:\\KYH\\02.PYTHON\\crawled_data\\'
    CHROME_DRIVER_PATH = 'D:\\KYH\\02.PYTHON\\vEnvDjango3_5_2_64\\'
    CRAWLING_URL = 'https://pccs.kepco.co.kr/iSmart/pccs/usage/getGlobalUsageStats.do?year={:s}&month={:s}&day={:s}&searchType_min=15'

    def __init__(self, search_year):
        self.search_year = search_year
        self.crawling_list = []
        self.data = {}
        self.set_chrome_driver()

    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(Ismart_crawler.CHROME_DRIVER_PATH + 'chromedriver.exe')

    def ismart_login(self):
        self.driver.get("https://pccs.kepco.co.kr/iSmart/jsp/cm/login/login.jsp")
        self.driver.find_element_by_name("userId").send_keys(Ismart_crawler.ID)
        self.driver.find_element_by_name("password").send_keys(Ismart_crawler.PW)
        self.driver.find_element_by_name("password").submit()
        time.sleep(3)

    def set_crawling_list(self):
        for month in range(1, 13):
            self.crawling_list.append([(str(self.search_year), str(month).rjust(2, '0'), str(day).rjust(2, '0')) for day in range(1, calendar.monthrange(self.search_year, month)[1]+1)])

    def data_to_file(self):
        with open(Ismart_crawler.FILE_PATH + "ismart1.txt", "w", encoding="utf-8") as file:
            print('데이터를 저장하는 중입니다.')
            for key, value in sorted(self.data.items(), key=operator.itemgetter(0)):
                file.write('======================================================\n')
                file.write('>> ' + key[0] + ' 년, ' + key[1] + ' 월, ' + key[2] + ' 일\n')
                for v in value:
                    file.write(','.join(v) + '\n')
                file.write('======================================================\n\n')
            print('데이터 저장이 완료되었습니다.')

    def get_ismart_data(self):
        for list_per_month in self.crawling_list:
            for day in list_per_month:
                self.driver.get(self.CRAWLING_URL.format(day[0], day[1], day[2]))
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                table_list = soup.find_all('table', class_='table02')  # 0 ~ 1 : 1시간 단위 요약 정보, 2 ~ 3 : 15분 단위 요약 정보

                for tr_tag in table_list[2].find_all('tr'):
                    if tr_tag.find('td', class_='bg_white_c'):
                        temp = []
                        if self.data.get((day[0], day[1], day[2])) is None:
                            self.data[(day[0], day[1], day[2])] = []
                        for td_tag in tr_tag.find_all('td'):
                            temp.append(td_tag.text)
                        self.data[(day[0], day[1], day[2])].append(temp)

                for tr_tag in table_list[3].find_all('tr'):
                    if tr_tag.find('td', class_='bg_white_c'):
                        temp = []
                        if self.data.get((day[0], day[1], day[2])) is None:
                            self.data[(day[0], day[1], day[2])] = []
                        for td_tag in tr_tag.find_all('td'):
                            temp.append(td_tag.text)
                        self.data[(day[0], day[1], day[2])].append(temp)

                time.sleep(random.randint(5, 10))

    def play_crawling(self):
        try:
            self.ismart_login()
            self.set_crawling_list()
            self.get_ismart_data()
            self.data_to_file()
            self.driver.quit()
        except Exception as e:
            self.data_to_file()
            print('종료 되었습니다.', e)

istart_crawler = Ismart_crawler(2016)
istart_crawler.play_crawling()
