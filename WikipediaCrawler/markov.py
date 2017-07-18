from WikipediaCrawler.database import Database
from random import randint

###################################################################################################
## Markov Chain 관련 클래스
###################################################################################################
class Markov(object):
    def __init__(self, keyword):
        Database.createConn()
        self.keyword = keyword
        self.db = Database(None, keyword)

    # 마르코프 이론을 통한 랜덤 문장 생성
    def markovMain(self):
        length = 100
        sentence = ''
        # 'shigeo': {'nagashima': [1, None], 'tanaka': [1, None], '？dachi': [1, None], 'toya': [1, None]}, 'idealism': {'in': [1, 'p'], 'was': [1, None], 'by': [2, 'p'], ',': [4, None], 'championing': [1, 'g,v'], 'of': [3, 'p'], '.': [2, None], 'and': [1, None]}
        markovDict = self.db.createMarkovDict()
        print(markovDict)
        currentWord = ''

        startWord = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']

        # 각 키워드별 가장 빈도수 높은 명사를 추출
        for key in sorted(markovDict, key=lambda k: sum(map(lambda v: v[0] if isinstance(v, list) else 0, markovDict[k].values())), reverse=True):
            if markovDict[key]['#type'] in startWord:
                currentWord = key
                break

        bol = True

        for i in range(0, length):
            sentence += currentWord+' '
            try:
                currentWord = self.retrieveRandomWord(markovDict[currentWord])
            except KeyError:
                print(self.keyword + '에 해당하는 문구가 존재하지 않습니다.')
                bol = False
                break

            if currentWord == '.':  # DB 에 각 라인별로 데이터가 들어가 있으므로, . 에 대한 다음 단어가 존재하지 않는다.
                sentence += '.'
                break

        if bol:
            print(self.keyword + ' : ' + sentence)

    # 랜덤으로 단어 검색
    def retrieveRandomWord(self, wordList):
        ranIndex = randint(1, self.wordListSum(wordList))

        for word, value in wordList.items():
            if word != '#type':  # 단어의 타입을 결정하는 KEY 라서 제외
                ranIndex -= value[0]
                if ranIndex <= 0:
                    return word

    # 특정 단어와 연관되는 단어들의 합
    def wordListSum(self, wordList):
        sum = 0
        for word, value in wordList.items():
            if word != '#type':
                sum += value[0]
        return sum