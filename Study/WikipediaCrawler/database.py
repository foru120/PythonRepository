import cx_Oracle
from WikipediaCrawler import common
import os
import time
from nltk import word_tokenize, pos_tag

os.environ["NLS_LANG"] = ".AL32UTF8"  # 오라클 데이터베이스에 특수 문자들도 넣기 위해 character set 변환

###################################################################################################
## Database 관련 클래스
###################################################################################################
class Database(object):

    # 클래스 변수
    connect_cnt = 0
    db_pool = None

    def __init__(self, base_url, keyword):
        self.base_url = base_url
        self.keyword = keyword

    @classmethod
    def createConn(cls):
        # 데이터베이스 연결 초기화
        if Database.db_pool is None:
            #######################################################################################
            ## Mysql DB 연동
            #######################################################################################
            # Database._linkConn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='mysql', db='wikipedia', charset='utf8')
            # Database._historyConn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='mysql', db='wikipedia', charset='utf8')
            # Database._fileConn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='mysql', db='wikipedia', charset='utf8')

            #######################################################################################
            ## Oracle DB 연동
            #######################################################################################
            # Database._linkConn = cx_Oracle.connect(common.USERNAME+'/'+common.PASSWD+'@localhost:'+common.PORT+'/'+common.SID)
            # Database._historyConn = cx_Oracle.connect(common.USERNAME+'/'+common.PASSWD+'@localhost:'+common.PORT+'/'+common.SID)
            # Database._fileConn = cx_Oracle.connect(common.USERNAME+'/'+common.PASSWD+'@localhost:'+common.PORT+'/'+common.SID)

            #######################################################################################
            ## Oracle DBCP 연동
            #######################################################################################
            Database.db_pool = cx_Oracle.SessionPool(user=common.USERNAME,
                                                     password=common.PASSWD,
                                                     dsn=common.SID,
                                                     min=10,
                                                     max=20,
                                                     increment=2,
                                                     homogeneous=True)
            print('데이터베이스 커넥션 객체가 생성되었습니다.')

    @classmethod
    def releaseConn(cls, cur, conn):
        cur.close()
        Database.db_pool.release(conn)

    @classmethod
    def getConnection(cls):
        Database.connect_cnt += 1
        return Database.db_pool.acquire()


    ###############################################################################################
    ## link_finder.py
    ###############################################################################################
    # URL 이 스크랩 되었는지 확인하는 함수
    def urlScraped(self, url):
        conn = Database.getConnection()
        cur = conn.cursor()

        cur.execute('select * from urls where keyword = :1 and urlname = :2', [self.keyword, url])
        page = cur.fetchone()

        if cur.rowcount == 0:
            Database.releaseConn(cur, conn)
            return False

        cur.execute('select * from links where keyword = :1 and tourlid = :2', [self.keyword, page[0]])
        cur.fetchone()

        if cur.rowcount == 0:
            Database.releaseConn(cur, conn)
            return False

        Database.releaseConn(cur, conn)
        return True

    # 스크랩 된 URL 저장하는 함수
    def insertUrlIfNotExists(self, urlName):
        conn = Database.getConnection()
        cur = conn.cursor()

        cur.execute('select * from urls where keyword = :1 and urlname = :2', [self.keyword, urlName])
        data = cur.fetchone()

        if cur.rowcount == 0:
            cur.execute('insert into urls(urlId, keyword, urlname) values(url_seq.nextval, :1, :2)', [self.keyword, urlName])
            conn.commit()
            cur.execute('select url_seq.currval from dual')
            urlId = cur.fetchone()[0]

            Database.releaseConn(cur, conn)
            return urlId
        else:
            Database.releaseConn(cur, conn)
            return data[0]

    # 스크랩된 URL 과 해당 URL 이 링크된 URL 저장하는 함수
    def insertLink(self, fromUrlId, toUrlId):
        conn = Database.getConnection()
        cur = conn.cursor()

        cur.execute('select * from links where keyword = :1 and fromurlid = :2 and tourlid = :3', [self.keyword, fromUrlId, toUrlId])
        cur.fetchone()

        if cur.rowcount == 0:
            cur.execute('insert into links(keyword, fromurlid, tourlid) values(:1, :2, :3)', [self.keyword, fromUrlId, toUrlId])
            conn.commit()

        Database.releaseConn(cur, conn)


    ###############################################################################################
    ## history_finder.py
    ###############################################################################################
    # 스크랩 된 URL 중 가장 마지막 URL 의 ID 를 리턴시키는 함수
    def getMaxUrlID(self):
        conn = Database.getConnection()
        cur = conn.cursor()

        cur.execute('select max(urlId) from urls where keyword = :1', [self.keyword])
        maxUrlId = cur.fetchone()[0]

        Database.releaseConn(cur, conn)
        return maxUrlId

    # 스크랩 된 URL 을 리턴시키는 함수
    def getUrlName(self, startUrlId):
        conn = Database.getConnection()
        cur = conn.cursor()

        cur.execute('select urlId, urlName from urls where keyword = :1 and urlId <= :2', [self.keyword, startUrlId])
        urlList = cur.fetchall()

        Database.releaseConn(cur, conn)
        return urlList

    # 스크랩 된 URL 의 History 내역을 저장하는 함수
    def insertHistory(self, history_list):
        conn = Database.getConnection()
        cur = conn.cursor()

        for history in history_list:
            cur.execute('insert into history values(:1, :2, :3, :4, :5, :6, :7, :8, :9, :10)',
                        [history['urlId'], history['historyTime'], history['userLink'], history['countryName'], history['city'],
                         history['latitude'], history['longitude'], history['isMember'], history['historySize'], history['plusMinus']])
        conn.commit()

        Database.releaseConn(cur, conn)

        print(str(history_list[0]['urlId'])+' 의 History 내역이 DB 로 저장되었습니다.')

    def getLatLong(self):
        conn = Database.getConnection()
        cur = conn.cursor()

        cur.execute('select distinct latitude, longitude from urls a, history b where a.urlid = b.urlid and a.keyword = :1 and latitude is not null and longitude is not null', [self.keyword])
        pos_list = cur.fetchall()
        conn.commit()

        Database.releaseConn(cur, conn)
        return pos_list


    ###############################################################################################
    ## file_io.py
    ###############################################################################################
    def insertFileToDB(self, content_list):
        conn = Database.getConnection()
        cur = conn.cursor()
        now = time.localtime()
        print('== '+content_list[0][1]+' : 데이터를 DB 로 이관 중 ('+str(now.tm_year)+'/'+str(now.tm_mon)+'/'+str(now.tm_mday)+' '+
              str(now.tm_hour)+':'+str(now.tm_min)+':'+str(now.tm_sec)+')')

        try:
            #######################################################################################
            ## Oracle Batch I/O
            #######################################################################################
            # cur.prepare('insert into crawledfiles values(:1, :2, :3)')
            # cur.executemany(None, content_list)

            for content in content_list:
                cur.execute('insert into crawledfiles values(:1, :2, :3)', [content[0], content[1], content[2]])
            conn.commit()

            Database.releaseConn(cur, conn)
        except Exception as e:
            print(e)

        print('== '+content_list[0][1]+' : DB로 데이터 이관이 완료되었습니다. ('+str(now.tm_year)+'/'+str(now.tm_mon)+'/'+str(now.tm_mday)+' '+
              str(now.tm_hour)+':'+str(now.tm_min)+':'+str(now.tm_sec)+')')


    ###############################################################################################
    ## markov.py
    ###############################################################################################
    # 마르코프 단어 사전 생성
    def createMarkovDict(self):
        markovDict = {}
        conn = Database.getConnection()
        cur = conn.cursor()

        ###########################################################################################
        ## NLTK 적용 전
        ###########################################################################################
        # wordDict = {}
        # cur.execute("select lower(word), listagg(gubun, ',') within group(order by gubun) gubun from wordlist group by lower(word)")
        # for row in cur.fetchall():
        #     wordDict[row[0]] = row[1]
        #
        # cur.execute('select lower(content) from crawledfiles where keyword = :1', [keyword])
        # rows = cur.fetchall()
        #
        # for row in rows:
        #     words = self.cleanText(row[0])
        #     for i in range(1, len(words)):
        #         if words[i-1] not in markovDict:
        #             markovDict[words[i-1]] = {}
        #
        #         if words[i] not in markovDict[words[i-1]]:  # 현재 단어에서 다음 단어로의 데이터가 없는 경우
        #             if words[i].lower() in wordDict.keys():
        #                 markovDict[words[i - 1]][words[i]] = [0, wordDict[words[i].lower()]]
        #             else:
        #                 markovDict[words[i - 1]][words[i]] = [0, None]
        #         markovDict[words[i - 1]][words[i]][0] += 1

        ###########################################################################################
        ## NLTK 적용 후
        ###########################################################################################
        cur.execute("select regexp_replace(content, '[^a-zA-Z0-9\.,;:?()[:space:]]+', '') content from crawledfiles")

        for row in cur.fetchall():
            words = pos_tag(word_tokenize(row[0]))
            for i in range(1, len(words)):
                if words[i-1][0] not in markovDict.keys():
                    markovDict[words[i-1][0]] = {'#type': words[i-1][1]}

                if words[i][0] not in markovDict[words[i-1][0]]:
                    markovDict[words[i-1][0]][words[i][0]] = [0, words[i][1]]

                markovDict[words[i-1][0]][words[i][0]][0] += 1

        Database.releaseConn(cur, conn)

        return markovDict

    # 문자열 정리하는 함수
    def cleanText(self, row):
        # 줄바꿈 문자와 따옴표 제거
        row = row.replace('\n', ' ').replace('\"', '')

        # 구두점 단어로 취급
        for symbol in [',', '.', ';', ':']:
            row = row.replace(symbol, ' ' + symbol + ' ')

        # 빈 단어 제거
        words = row.split(' ')
        words = [word for word in words if word != '']

        return words