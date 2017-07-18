import os
import re
import time
import threading
import csv

from WikipediaCrawler.database import Database

###################################################################################################
## File 관련 클래스
###################################################################################################
class File(object):

    tot_file_count = 0

    def __init__(self, keyword):
        self.keyword = keyword
        self.db = Database(None, self.keyword)

    # 크롤링 된 page 들을 모아놓을 directory 생성하는 함수
    @classmethod
    def create_directory(self, directory_name):
        if not os.path.exists(directory_name):
            print('Creating Directory : ' + directory_name)
            os.makedirs(directory_name)

    # 크롤링 된 URL 을 기록하기 위한 File 생성하는 함수
    @classmethod
    def create_crawled_files(self, directory_name):
        crawled = directory_name + '/crawled.txt'

        if not os.path.isfile(crawled):
            self.write_file(crawled, '')

    # 처음으로 파일 생성하는 함수
    @classmethod
    def write_file(self, path, data):
        with open(path, 'w', encoding='utf-8') as file:
            file.write(data)

    # 기존 파일에 내용 추가하는 함수
    @classmethod
    def append_to_file(self, path, data):
        with open(path, 'a', encoding='utf-8') as file:
            file.write(data+'\n')

    # 파일을 읽어서 set 형태로 변환하는 함수
    @classmethod
    def file_to_set(self, file_name):
        results = set()
        with open(file_name, 'rt') as file:
            for line in file:
                results.add(line.replace('\n', ''))  # 개행문자(\n)를 ''로 변환
        return results

    ###################################################################################################
    ## file_thread 부분
    ###################################################################################################
    # 파일을 읽어서 한 줄씩 DB 에 저장하는 함수
    def file_to_db(self):
        inserted_files = []
        out_cnt = 0

        while True:
            scraped_files = os.listdir('ScrapedFiles/')
            filtered_files = []
            for file in scraped_files:
                if self.keyword+'#' in file:
                    filtered_files.append(file)

            scraped_files = filtered_files

            if len(scraped_files) == 0:
                time.sleep(1)
                continue

            for inserted_file in inserted_files:
                if inserted_file in scraped_files:
                    scraped_files.remove(inserted_file)

            print('스크랩 대상 파일 개수 : ' + str(len(scraped_files)))

            if len(scraped_files) == 0 and threading.activeCount() < 3:
                print('=========================================================================')
                print('활성중인 쓰레드가 없거나, 스크랩 대상 파일이 없습니다.')
                print('=========================================================================')
                if out_cnt > 2:
                    break
                else:
                    time.sleep(3)
                    out_cnt += 1
                    continue

            file_cnt = len(scraped_files)
            thread_list = []
            index = 0

            while True:
                cnt = 0
                for i in range(0, file_cnt if file_cnt <= 1 else 1):
                    thread_list.append(threading.Thread(target=self.fileThread, args=(scraped_files[index].replace(self.keyword+'#', ''))))
                    inserted_files.append(scraped_files[index])
                    index += 1
                    cnt += 1

                # thread 수행
                for thread in thread_list:
                    thread.start()

                # 전체 thread 수행 완료까지 대기
                for thread in thread_list:
                    thread.join()

                thread_list.clear()
                file_cnt -= cnt

                if file_cnt == 0:
                    break

            out_cnt = 0

            time.sleep(1)

        File.tot_file_count = 0

    def fileThread(self, filename):
        print(filename, self.keyword)
        text = ''
        with open('ScrapedFiles/' + self.keyword + '#' + filename, 'rt', encoding='utf-8') as file:
            for line in file:
                text += line.strip()  # 문자열 양쪽 특정 문자 제거(SQL trim 함수와 동일)

        #######################################################################################
        # 한글 문자열 라인별 출력
        text = re.sub('\[\d+\]', '', re.sub('오[\.]', '오.\n', re.sub('다[\.]', '다.\n', text)))

        text_list = []
        for t in re.findall('^.+[다|오]\.$', text, re.MULTILINE):
            text_list.append((self.keyword, filename, t.strip()))
        #######################################################################################

        #######################################################################################
        # 영문 문자열 라인별 출력
        # text = re.sub('\[\d+\]', '', re.sub('\.', '.\n', text))
        #
        # text_list = []
        # for t in re.findall('^.+\.$', text, re.MULTILINE):
        #     text_list.append((self.keyword, filename, t.strip()))
        #######################################################################################

        print('keyword : ' + self.keyword + ', file count : ' + str(File.tot_file_count) + ', filename : ' + filename)
        self.db.insertFileToDB(text_list)
        File.tot_file_count += 1


    ###################################################################################################
    ## 각종 확장자 파일 저장&추출
    ###################################################################################################
    # html table 정보를 csv 파일로 저장
    def tableToFile(self, html_table):
        rows = html_table.file_all('tr')

        with open('downloads/editors.csv', 'wt', encoding='utf-8') as csvFile:
            writer = csv.writer(csvFile)
            for row in rows:
                csvRow = []
                for cell in row.find_all(['td', 'th']):
                    csvRow.append(cell.get_text())
                writer.writerow(csvRow)
