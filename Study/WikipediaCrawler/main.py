import threading
import time

from WikipediaCrawler.database import Database
from WikipediaCrawler.link_finder import Link
from WikipediaCrawler.history_finder import History
from WikipediaCrawler.file_io import File
from WikipediaCrawler import common

if __name__ == '__main__':
    Database.createConn()
    File.create_directory('ScrapedFiles')

    for keyword in common.KEYWORD:
        # 해당 키워드에 대한 링크 수집
        # o_link = Link(common.ENG_BASE_URL, keyword)
        # o_link.getLinks('/wiki/'+keyword, 0)

        # 링크별 글 수집
        # o_file = File(keyword)
        # o_file.file_to_db()

        # 링크별 히스토리 수집
        o_history = History(common.ENG_BASE_URL, keyword)
        o_history.getHistory()

        print('======================================================')
        print(keyword + ' : 키워드에 대한 자료 조사가 종료되었습니다.')
        print('======================================================')

        # common.link_thread = threading.Thread(target=getLinks, args=(common.ENG_BASE_URL, '/wiki/'+keyword, 0, keyword))
        # common.history_thread = threading.Thread(target=getHistory, args=(common.ENG_BASE_URL, keyword))
        # common.file_thread = threading.Thread(target=file_to_db, args=(keyword,))

        # common.link_thread.start()
        # time.sleep(3)
        # common.history_thread.start()
        #time.sleep(3)
        # common.file_thread.start()

        # common.link_thread.join()
        # common.file_thread.join()
        # common.history_thread.join()
        #print('history_thread 종료')

