#-*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import time
from selenium import webdriver
import operator

class MelonCrawler(object):
    FILE_PATH = 'D:\\02.Python\\data\\'  # 크롤링 된 데이터가 저장되는 곳
    CHROME_DRIVER_PATH = 'D:\\02.Python\\'  # 크롬드라이버 위치
    GENRE = {'POP': (1955, 2016), 'KPOP': (1964, 2016)}  # 장르 종류 -> POP(시작년도, 종료년도) : 팝, KPOP : 가요(시작년도, 종료년도)
    SEPARATOR = ';'

    def __init__(self):
        self.chart_data = {}
        self.lyrics_data = {}
        self.chart_url = 'http://www.melon.com/chart/age/index.htm?chartGenre={}&chartDate={}'
        self.lyrics_url = 'http://www.melon.com/song/detail.htm?songId={}'
        self.set_chrome_driver()

    def set_chrome_driver(self):
        self.driver = webdriver.Chrome(MelonCrawler.CHROME_DRIVER_PATH + 'chromedriver')

    # 크롤링 된 데이터를 파일로 저장하는 함수
    def data_to_file(self, filename, mode, genre=None, year=None):
        if mode == 1:  # chart 데이터를 저장하는 경우
            with open(MelonCrawler.FILE_PATH + filename, "a", encoding="utf-8") as file:
                for key, values in sorted(self.chart_data.items(), key=operator.itemgetter(0)):
                    for value in values:
                        file.write(str(genre) + MelonCrawler.SEPARATOR + str(key) + MelonCrawler.SEPARATOR + str(value[0])
                                   + MelonCrawler.SEPARATOR + value[1] + MelonCrawler.SEPARATOR + value[2] + MelonCrawler.SEPARATOR + value[3] + '\n')
            self.chart_data.clear()
        else:  # 가사 데이터를 저장하는 경우
            for song_id, lyrics in self.lyrics_data.items():
                with open(MelonCrawler.FILE_PATH+genre+'_'+str(year)+'_'+str(song_id)+'.txt', 'a', encoding='utf-8') as file:
                    file.write(lyrics)
            self.lyrics_data.clear()

    # 장르별, 년도별, 1~50 순위권 내의 노래제목, 가수명, 가사ID 크롤링하는 함수
    def get_chart_data(self, genre, year):
        self.driver.get(self.chart_url.format(genre, year))
        time.sleep(5)
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        rank = 1
        temp_list = []
        tr_tags = soup.find('form', id='frm').find('table').find('tbody').find_all('tr', class_='lst50')

        for tr_tag in tr_tags:
            td_tag = tr_tag.find('td', class_='t_left')
            songId = re.match('.*\'([0-9]+)\'.*', td_tag.find('a', class_='btn btn_icon_detail')['onclick']).group(1)
            if td_tag.find('div', class_='ellipsis rank01').find('a') is None:
                songTitle = td_tag.find('div', class_='ellipsis rank01').find('div', class_='ellipsis').text
            else:
                songTitle = td_tag.find('div', class_='ellipsis rank01').find('a').text
            songTitle = songTitle.strip('\n').strip().replace('\n', ' ')
            singer = td_tag.find('div', class_='ellipsis rank02').find('span', class_='checkEllipsis').text.strip()
            temp_list.append((rank, songTitle, singer, songId))
            rank += 1

        self.chart_data[year] = temp_list  # {'1950': [(1, 노래제목, 가수명, songId) ...], '1951': [(2, 노래제목, 가수명, songId) ...] ...} -> 데이터 형태

    # 장르별, 년도별, 가사 크롤링 하는 함수
    def get_lyrics_data(self, genre, year, song_ids):
        for song_id in song_ids:
            self.driver.get(self.lyrics_url.format(song_id))
            time.sleep(3)
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            if soup.find('div', id='d_video_summary') is not None:
                lyrics = soup.find('div', id='d_video_summary')
                lyrics = re.findall('^.*$', str(lyrics).strip(), re.M)[1].strip().replace('<br/>', '\n')
                self.lyrics_data[song_id] = lyrics
        self.data_to_file(None, 2, genre, year)

    # 저장된 song id 를 파일로부터 가져오는 함수
    def get_song_id(self):
        song_data = {}
        with open(MelonCrawler.FILE_PATH+'chart_data.txt', 'rt', encoding='utf-8') as f:
            for line in iter(f):
                data = line.split(MelonCrawler.SEPARATOR)
                if song_data.get(data[0]) is None:
                    song_data[data[0]] = {}
                if song_data[data[0]].get(int(data[1])-int(data[1])%10) is None:
                    song_data[data[0]][int(data[1])-int(data[1])%10] = set()
                song_data[data[0]][int(data[1]) - int(data[1]) % 10].add(int(data[-1].strip('\n')))
        return song_data

    # 크롤링 수행하는 메인 함수
    def play_crawling(self):
        for genre, value in MelonCrawler.GENRE.items():  # chart 데이터 수집하는 부분
            print(genre+' 장르에 대한 chart 데이터 크롤링을 시작합니다.')
            for year in range(value[0], value[1]+1):
                self.get_chart_data(genre, year)
                if (year % 10) == 0:  # 중간 저장
                    self.data_to_file('chart_data.txt', 1, genre)
            self.data_to_file('chart_data.txt', 1, genre)
            print(genre+' 장르에 대한 chart 데이터 크롤링이 완료되었습니다.')

        song_data = self.get_song_id()

        for genre, values in song_data.items():
            print(genre+' 장르에 대한 가사 크롤링을 시작합니다.')
            for year, song_ids in values.items():
                self.get_lyrics_data(genre, year, song_ids)
            print(genre + ' 장르에 대한 가사 크롤링이 완료되었습니다.')

crawler = MelonCrawler()
crawler.play_crawling()