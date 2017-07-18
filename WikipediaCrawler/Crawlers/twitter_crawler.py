from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os


# def get_save_path():
#     save_path = input("Enter the file name and file location :" )
#     save_path = save_path.replace("\\", "/")
#     if not os.path.isdir(os.path.split(save_path)[0]):
#         os.mkdir(os.path.split(save_path)[0])
#     return save_path

file = open('D:\\KYH\\02.PYTHON\\crawled_data\\test.txt', 'w', encoding='utf-8')

binary = 'D:\\KYH\\02.PYTHON\\vEnvDjango3_5_2_64\\chromedriver.exe'
browser = webdriver.Chrome(binary)
browser.get("https://twitter.com/search-home")
elem = browser.find_element_by_id("search-home-input")

elem.send_keys("삼성전자")
elem.submit()

for i in range(1,10):
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    time.sleep(5)

time.sleep(5)
html = browser.page_source  # 내가 브라우져로 보고있는 소스를 볼려고하는것이다.
                            # 그런데 그냥 열면 사용자가 end 버튼틀 눌러서 컨트롤
                            # 한게 반영 안된것이 열린다.
soup = BeautifulSoup(html, "html.parser")
tweet_tag = soup.find_all(class_="tweet-text")

for i in tweet_tag:
    tweet_text = i.get_text(strip=True)
    print(tweet_text)
    file.write(tweet_text)

file.close()
browser.quit()