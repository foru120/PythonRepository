#-*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from datetime import datetime
import operator

class FacebookCrawler:
    # FILE_PATH = 'D:\\02.Python\\facebook_data\\'
    # CHROME_DRIVER_PATH = 'D:\\02.Python\\'
    FILE_PATH = 'D:\\KYH\\02.PYTHON\\facebook_data\\'
    CHROME_DRIVER_PATH = 'C:\\Users\\stu\\Downloads\\chromedriver_win32\\'

    def __init__(self, searchKeyword, startMonth, endMonth, scroll_down_cnt):
        self.searchKeyword = searchKeyword
        self.startMonth = startMonth
        self.endMonth = endMonth
        self.scroll_down_cnt = scroll_down_cnt
        self.data = {}
        self.url = 'https://www.facebook.com/search/str/' + searchKeyword + '/keywords_top?filters_rp_creation_time=%7B"start_month%22%3A"' + startMonth + '"%2C"end_month"%3A"' + endMonth + '"%7D'
        self.set_chrome_driver()

    # chrome driver 생성 후 chrome 창 크기 설정하는 함수.
    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(FacebookCrawler.CHROME_DRIVER_PATH + 'chromedriver.exe')
        self.driver.set_window_size(1024, 768)

    # facebook 홈페이지로 이동 후 email, password 를 입력하고 submit 보내는 함수. (로그인)
    def facebook_login(self):
        self.driver.get("https://www.facebook.com/")
        self.driver.find_element_by_id("email").clear()
        self.driver.find_element_by_id("email").send_keys("pyth12312@gmail.com")
        self.driver.find_element_by_id("pass").clear()
        self.driver.find_element_by_id("pass").send_keys("Python123")
        self.driver.find_element_by_id("pass").submit()
        time.sleep(5)
        self.driver.get(self.url)

    # facebook page scroll down 하는 함수
    def page_scroll_down(self):
        for i in range(1, self.scroll_down_cnt):
            self.driver.find_element_by_xpath("//body").send_keys(Keys.END)
            time.sleep(3)

    # 크롤링 된 데이터를 파일로 저장하는 함수
    def data_to_file(self):
        with open(FacebookCrawler.FILE_PATH + self.searchKeyword + ".txt", "w", encoding="utf-8") as file:
            print('데이터를 저장하는 중입니다.')
            for key, value in sorted(self.data.items(), key=operator.itemgetter(0)):
                file.write(str(datetime.fromtimestamp(key)) + ' : ' + value + '\n')
            file.close()
            print('데이터 저장이 완료되었습니다.')

    # 크롤링 수행하는 메인 함수
    def play_crawling(self):
        try:
            self.facebook_login()
            time.sleep(5)
            self.page_scroll_down()

            html = self.driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup.select('.fbUserContent._5pcr'):
                usertime = tag.find('abbr', class_='_5ptz')
                content = tag.find('div', class_='_5pbx userContent').find('p')
                if usertime is not None and content is not None:
                    self.data[int(usertime['data-utime'])] = content.get_text(strip=True)

            self.data_to_file()
            self.driver.quit()
        except Exception as e:
            print('정상 종료 되었습니다.', e)

crawler = FacebookCrawler('평창올림픽', '2014-02', '2014-02', 5)
crawler.play_crawling()