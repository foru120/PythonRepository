import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

binary = 'D:\\KYH\\02.PYTHON\\vEnvDjango3_5_2_64\\chromedriver.exe'
browser = webdriver.Chrome(binary)
browser.get("https://www.google.co.kr")
elem = browser.find_element_by_id("lst-ib")
# find_elements_by_class_name("")

# 검색어 입력
elem.send_keys("아이언맨")
elem.submit()

# 반복할 횟수
for i in range(1, 2):
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    time.sleep(5)

time.sleep(5)
html = browser.page_source
soup = BeautifulSoup(html, "html.parser")


# print(soup)
# print(len(soup))

def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_="_img")
    for im in imgList:
        params.append(im["src"])
    return params


def fetch_detail_url():
    params = fetch_list_url()
    # print(params)
    a = 1
    for p in params:
        # 다운받을 폴더경로 입력
        urllib.request.urlretrieve(p, "D:\\KYH\\02.PYTHON\\crawled_image\\" + str(a) + ".jpg")

        a = a + 1


fetch_detail_url()

browser.quit()