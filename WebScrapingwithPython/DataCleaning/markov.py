from urllib.request import urlopen
from random import randint

def wordListSum(wordList):
    sum = 0
    for word, value in wordList.items():
        sum += value
    return sum

def retrieveRandomWord(wordList):
    randIndex = randint(1, wordListSum(wordList))
    for word, value in wordList.items():
        randIndex -= value
        if randIndex <= 0:
            return word

def buildWordDict(text):
    # 줄바꿈 문자와 따옴표를 제거합니다.
    text = text.replace('\n', ' ')
    text = text.replace('\"', '')

    # 구두점 역시 단어로 취급해서 마르코프 체인에 들어가게 합니다
    punctuation = [',', '.', ';', ':']
    for symbol in punctuation:
        text = text.replace(symbol, ' '+symbol+' ')

    words = text.split(' ')

    # 빈 단어를 제거합니다
    words = [word for word in words if word != '']

    wordDict = {}

    for i in range(1, len(words)):
        if words[i-1] not in wordDict:
            # 이 단어에 필요한 새 딕셔너리를 만듭니다
            wordDict[words[i-1]] = {}
        if words[i] not in wordDict[words[i-1]]:
            wordDict[words[i-1]][words[i]] = 0
        wordDict[words[i-1]][words[i]] = wordDict[words[i-1]][words[i]]+1

    return wordDict

text = str(urlopen('http://pythonscraping.com/files/inaugurationSpeech.txt').read(), 'utf-8')
wordDict = buildWordDict(text)

# 길이가 100인 마르코프 체인을 생성합니다.
length = 100
chain = ''
currentWord = 'I'

for i in range(0, length):
    chain += currentWord+' '
    currentWord = retrieveRandomWord(wordDict[currentWord])

print(chain)