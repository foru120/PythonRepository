import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
from selenium.webdriver.common.keys import Keys
import time                     # 중간중간 sleep 을 걸어야 해서 time 모듈 import
from selenium.webdriver.chrome.options import Options

########################### 크롬창 안뜨게 하기 ###########################
# chrome_options = Options()
# chrome_options.add_argument('--headless')

########################### url 받아오기 ###########################

# 웹브라우져로 크롬을 사용할거라서 크롬 드라이버를 다운받아 위의 위치에 둔다
# 팬텀 js로 하면 백그라운드로 실행할 수 있음
binary = 'D:/utils/chromedriver.exe'

browser = webdriver.Chrome(binary)

browser.get("http://cyberbureau.police.go.kr/prevention/sub7.jsp?mid=020600")
browser.find_element_by_id("idsearch").send_keys('01071896576')
browser.find_element_by_id("getXmlSearch").click()

time.sleep(1)

soup = BeautifulSoup(browser.page_source, 'html.parser')

resultTag = soup.find('p', id='search_result', class_='fraud_box')

print(resultTag.text)