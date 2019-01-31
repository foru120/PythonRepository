from bs4 import BeautifulSoup
from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
import time                           # 중간중간 sleep 을 걸어야 해서 time 모듈 import

# 웹브라우져로 크롬을 사용할거라서 크롬 드라이버를 다운받아 위의 위치에 둔다
binary = 'd:\chromedriver/chromedriver.exe'

# 브라우져를 인스턴스화
browser = webdriver.Chrome(binary)

# 성범죄자 알림 e 성범죄자 찾아보기 URL 받아오기
browser.get("https://www.sexoffender.go.kr/m1s2_login.nsc")
time.sleep(3)

# 개인정보 활용에 대한 동의 팝업창 클릭
browser.find_element_by_xpath("//img[@id='agree_btn']").click()
time.sleep(3)

# 주민등록번호 인증 클릭
browser.find_element_by_xpath("//input[@value='4']").click()

# 이름 입력
browser.find_element_by_id("kname").send_keys("인증에 필요한 본인 이름 입력")

# 주민등록번호 입력
browser.find_element_by_id("socno1").send_keys("주민등록번호 앞 6자리 입력")
browser.find_element_by_id("socno2").send_keys("주민등록번호 뒤 6자리 입력")


# 주민등록번호 인증 클릭
browser.find_element_by_xpath("//input[@value='주민등록번호 인증']").click()
browser.get("https://www.sexoffender.go.kr/index.nsc")


try:
    # 성범죄자 찾아보기 조건으로 검색 URL 받아오기
    browser.get("https://www.sexoffender.go.kr/m1s2_3.nsc")
    time.sleep(3)

    # 조회할 이름 입력
    browser.find_element_by_id("textSearch").send_keys("민수")

    # 검색 버튼 클릭
    browser.find_element_by_xpath("//a[@class='btn_gray font']").click()

    # 조회 시 주민등록상 거주지 / 실제 거주지 텍스트 긁어오기(1페이지당 최대 5개)
    for i in range(1,6):
        for j in range(1,3):
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            result = soup.find_all('tr')[i]('td')[j]
            final = result.get_text(strip=True, separator='\n')
            if j == 1:
                print("주민등록상거주지 : ", final)
            else:
                print("실제 거주지 : ", final)
        print("===================================================")

except:
    print("검색조건의 인터넷 공개 대상자가 없습니다")

# 크롬 브라우저 닫기
browser.close()