# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from bs4 import BeautifulSoup
import time
import urllib.request
from PIL import Image
import os
import re
import matplotlib.image as mimage
import math
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys

class GoogleImageCrawler:
    __CHROME_DRIVER_PATH = 'D:\\03_utils\\'
    __IMAGE_PATH = ['D:\\01_study\\99.DeepLearning\\03.Dataset\\dcgan\\Gogh\\']
    __DATA_PATH = ['D:\\01_study\\99.DeepLearning\\03.Dataset\\dcgan\\Gogh\\']

    __SEARCH_DOG = ['퍼그', '시베리안 허스키', '아메리칸 핏불 테리어', '래브라도 리트리버', '비글', '로트바일러', '시추', '도베르만 핀셔', '저먼 셰퍼드', '복서',
                    '차우차우', '푸들', '포메라니안', '불도그', '요크셔 테리어', '잉글리쉬 마스티프',
                    '샤페이', '그레이트 데인', '닥스훈트', '몰티즈', '치와와', '펨브록 웰시코기', '시바견', '아키타', '세인트버나드', '미니어처 핀셔', '페키니즈',
                    '비즐라', '잭 러셀 테리어', '그레이하운드', '래브라두들', '와이머라너',
                    '하바나 실크 독', '그레이트 피레니즈', '사모예드', '불마스티프', '코카푸', '뉴펀들랜드', '카발리에 킹 찰스 스패니얼', '잉글리시 코커 스패니얼',
                    '웨스트 하일랜드 화이트 테리어', '알래스칸 맬러뮤트', '오스트레일리언 셰퍼드',
                    '달마티안', '레온베르거', '휘핏', '스타포드셔불 테리어', '오스트레일리안 캐틀독', '캉갈 도그', '아메리칸 스태퍼드셔 테이러', '바센지']
    __SEARCH_CAT = ['브리티시 쇼트헤어', '러시안 블루', '벵골', '래그돌', '페르시안', '샴', '스핑크스', '메인쿤', '먼치킨', '터키시 앙고라', '버먼', '시베리안 고양이',
                    '스코티시 폴드', '아메리칸 쇼트헤어', '아비시니안', '엑조틱 쇼트헤어',
                    '토이거', '네벨룽', '샤르트뢰', '아메리칸 컬', '노르웨이 숲 고양이', '오리엔탈 쇼트헤어', '라팜', '사바나', '픽시 밥', '피터볼드', '버미즈',
                    '히말라얀', '이집션 마우', '봄베이', '쵸시', '싱가푸라', '맹크스', '소말리 고양이',
                    '오시캣', '코니시 렉스', '라가머핀', '발리니즈', '반고양이', '톤키니즈', '셀커크 렉스', '코랏', '유러피안 숏 헤어', '스노우슈', '티파니',
                    '아메리칸 밥테일', '타이캣', '재패니즈 밥테일', '아메리칸 와이어헤어', '터키시 반', '자바니즈']
    __SEARCH_IMAGE = ['고흐 별이 빛나는 밤', '고흐 붓꽃', '고흐 감자 먹는 사람들', '고흐 아를의 침실', '고흐 밤의 카페 테라스', '고흐 삼나무가 있는 밀밭',
                      '고흐 노란집', '고흐 아몬드 꽃', '고흐 까마귀가 나는 밀밭', '고흐 아를의 붉은 포도밭', '고흐 슈케베닌겐 바다 전경', '고흐 자화상',
                      '고흐 가셰 박사의 초상', '고흐 밤의 카페', '고흐 론강의 별이 빛나는 밤에', '고흐 귀에 붕대를 감은 자화상', '고흐 붉은 장미가 있는 꽃병',
                      '고흐 나무뿌리', '고흐 자포네제리', '고흐 탕기 영감의 초상', '고흐 오베르의 교회', '고흐 양비귀꽃', '고흐 영원의 문', '고흐 별이 빛나는 밤의 사이프러스',
                      '고흐 담배 피는 해골', '고흐 몽마르주의 일몰', '고흐 오베르의 집들', '고흐 아를의 만발한 포도나무 풍경', '고흐 고호의 어머니 초상화',
                      '고흐 도비니의 정원', '고흐 양귀비꽃', '고흐 오베르의 농촌풍경', '고흐 슬픔', '고흐 라 무스메', '고흐 자장가(룰랭부인)', '고흐 프로방스의 농가',
                      '고흐 네덜란드의 꽃밭', '고흐 레잘리스캉', '고흐 사이프러스 나무가 있는 녹색 밀밭', '고흐 분홍 장미', '고흐 에텐 정원의 추억',
                      '고흐 탬버린의 여인', '고흐 게 두 마리', '고흐 15송이 해바라기가 있는 꽃병', '고흐 닥터 펠렉스 레이 초상화', '고흐 가죽 나막신',
                      '고흐 씨 뿌리는 사람', '고흐 오후 : 휴식(밀레 모작)']
    __SEARCH_KEYWORD = [__SEARCH_IMAGE]

    def __init__(self):
        self.__image_urls = []  # 이미지를 다운받을 URL 주소.
        self.__image_data = []  # 이미지가 Gray Scale 로 변환된 데이터.
        self.__number = 1  # 이미지 번호.
        self.__keyword_cnt = len(GoogleImageCrawler.__SEARCH_KEYWORD)  # 검색 종류 개수.
        self._google_image_url = 'https://www.google.com/imghp?hl=ko'
        self.__rgb_cnt = 0
        self._set_chrome_driver()

    def _set_chrome_driver(self):
        '''
            chrome driver 설정하는 함수.
        '''
        self.driver = webdriver.Chrome(GoogleImageCrawler.__CHROME_DRIVER_PATH + 'chromedriver')

    def _extract_image_url(self, images):
        '''
            html 파일로부터 Image URL 정보를 추출하는 함수.
        '''
        for image in images:
            try:
                image_src = image['src']
                if image_src is not None:
                    self.__image_urls.append((self.__number, image_src))
                    self.__number += 1
            except KeyError:
                print(image['name'] + ', src 속성이 존재하지 않습니다.')

    def _get_image_crawling(self, keyword):
        '''
            특정 키워드에 대한 이미지 검색 후 검색된 이미지들의 URL 주소를 수집하는 함수.
        '''
        self.driver.get(self._google_image_url)
        self.driver.find_element_by_id("lst-ib").clear()
        self.driver.find_element_by_id("lst-ib").send_keys(keyword)
        self.driver.find_element_by_id("lst-ib").submit()
        time.sleep(3)

        before_img_cnt = 0
        clicked = False

        while True:
            self.driver.find_element_by_xpath("//body").send_keys(Keys.END)
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            images = soup.find_all('img', class_='rg_ic rg_i')
            after_img_cnt = len(images)

            if before_img_cnt == after_img_cnt:
                if soup.find('input', id='smb') and clicked is False:
                    try:
                        self.driver.find_element_by_id('smb').click()
                        clicked = True
                        time.sleep(3)
                    except:
                        clicked = True
                        time.sleep(3)
                else:
                    self._extract_image_url(images)
                    break
            else:
                before_img_cnt = after_img_cnt
                time.sleep(3)

    def _image_downloads(self, keyword):
        '''
            수집된 이미지 경로를 가지고 직접 이미지를 다운로드하는 함수.
        '''
        for name, url in self.__image_urls:
            urllib.request.urlretrieve(url, GoogleImageCrawler.__IMAGE_PATH[self.__curr_index] + keyword + '_' + str(name) + '.jpg')

    def _image_to_thumbnail(self):
        '''
            기존 원본 이미지를 특정 사이즈 형식으로 Thumbnail 을 수행하는 함수.
            이미지가 저장된 폴더로부터 이미지를 로드 후 썸네일 이미지 생성.
        '''
        size = (64, 64)
        for index in range(self.__keyword_cnt):
            for file in [filename for filename in os.listdir(GoogleImageCrawler.__IMAGE_PATH[index]) if
                         re.match('[0-9]+\.(jpg|jpeg|png)', filename) is not None]:
                try:
                    print(file)
                    filename, ext = os.path.splitext(file)

                    new_img = Image.new("RGB", (64, 64), "white")
                    im = Image.open(GoogleImageCrawler.__IMAGE_PATH[index] + str(file))
                    im.thumbnail(size, Image.ANTIALIAS)
                    load_img = im.load()  # (199, 129, 49)
                    load_newimg = new_img.load()
                    i_offset = (64 - im.size[0]) / 2
                    j_offset = (64 - im.size[1]) / 2

                    for i in range(0, im.size[0]):
                        for j in range(0, im.size[1]):
                            load_newimg[i + i_offset, j + j_offset] = load_img[i, j]

                    if ext.lower() in ('.jpeg', '.jpg'):
                        new_img.save(GoogleImageCrawler.__DATA_PATH[index] + str(filename) + '_64x64.jpeg')
                    elif ext.lower() == '.png':
                        new_img.save(GoogleImageCrawler.__DATA_PATH[index] + str(filename) + '_64x64.png')
                except Exception as e:
                    print(str(file), e)

    def _rgb2gray(self, rgb):
        '''
            YCrCb : 디지털(CRT, LCDl, PDP 등)을 위해서 따로 만들어둔 표현방법.
             - Y : Red*0.2126 + Green*0.7152 + Blue*0.0722
            YPbPr : 아날로그 시스템을 위한 표현방법.
             - Y : Red*0.299 + Green*0.587 + Blue*0.114
            실제 RGB 값들을 Gray Scale 로 변환하는 함수 .
        '''
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

        return np.array(gray).astype('int32')

    def _extract_rgb_from_image(self):
        '''
            크롤링 된 이미지 파일들을 읽어들여 Gray Scale 로 변환하는 함수.
        '''
        for index in range(self.__keyword_cnt):
            for name in [filename for filename in os.listdir(GoogleImageCrawler.__IMAGE_PATH[index]) if
                         re.search('[0-9]+\_128x128.+', filename) is not None]:
                try:
                    img = mimage.imread(GoogleImageCrawler.__IMAGE_PATH[index] + str(name))
                    gray = self._rgb2gray(img)
                    self.__image_data.append([gray, index, name])
                except OSError as e:
                    print(str(name) + ', 이미지를 식별할 수 없습니다.', e)
                    continue

                self.__rgb_cnt += 1
                if self.__rgb_cnt % 1000 == 0:
                    self._data_to_file(index)
                    self.__image_data.clear()

            self._data_to_file(index)
            self.__image_data.clear()
            self.__rgb_cnt = 0

    def _data_to_file(self, index):
        '''
            Gray Scale 로 변환된 이미지 정보를 파일로 기록하는 함수.
        '''
        print('데이터를 저장하는 중입니다.')
        for data in self.__image_data:
            x_shape, y_shape = data[0].shape
            temp_data = ''
            for x in range(0, x_shape - 1):
                for y in range(0, y_shape - 1):
                    if x == 0 and y == 0:
                        temp_data += str(data[0][x][y])
                    else:
                        temp_data += ',' + str(data[0][x][y])
            # temp_data += ',' + str(data[1])  # label

            with open(GoogleImageCrawler.__DATA_PATH + 'image_data_' + str(
                    math.ceil(self.__rgb_cnt / 1000)) + '_' + str(index) + '.csv', 'a', encoding='utf-8') as f:
                f.write(temp_data + '\n')
        print('데이터 저장이 완료되었습니다.')

    def play_crawler(self):
        '''
            이미지 크롤링에 필요한 함수들을 수행하는 함수.
        '''
        for index in range(0, self.__keyword_cnt):
            self.__curr_index = index
            for keyword in GoogleImageCrawler.__SEARCH_KEYWORD[self.__curr_index]:
                print('crawling start.')
                self._get_image_crawling(keyword)
                print('crawling complete.')
                print('image count : ' + str(len(self.__image_urls)))

                print('image downloading.')
                self._image_downloads(keyword)
                print('image downloading complete.')

                self.__image_urls.clear()

        # print('image to thumbnail start.')
        # self._image_to_thumbnail()
        # print('image to thumbnail end.')

        # print('rgb to gray start.')
        # self._extract_rgb_from_image()
        # print('rgb to gray end.')

        # self.driver.quit()


crawler = GoogleImageCrawler()
crawler.play_crawler()